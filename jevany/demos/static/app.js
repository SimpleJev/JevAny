"use strict";
const $ = id => document.getElementById(id);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const label = name => {
  const move = /^([xyz])_(plus|minus)_(1|5)cm$/.exec(name);
  return move ? `${move[1].toUpperCase()} ${move[2] === "plus" ? "+" : "−"}${move[3]} cm` :
    ({do:"Interact", noop:"Wait"}[name] || name.replaceAll("_", " "));
};
let config, link = {}, current = "doom", mode = "replay", replay, state, index = 0;
let playing = false, automatic = false, busy = false, generation = 0, loop = 0, actions = {};
let failedAttempt = false;
const EXAMPLE = {
  state: "A customer writes: I was charged twice for order 4182, and the second charge is still pending.",
  question: "Which team should handle this ticket?",
  options: [
    {name: "billing", description: "Payment problems, duplicate charges and refunds"},
    {name: "shipping", description: "Delivery problems and missing parcels"},
    {name: "accounts", description: "Sign-in and account ownership problems"},
  ],
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || `Request failed (${response.status})`);
  return result;
}
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
  if (message) { playing = false; automatic = false; }
}
function controls() {
  $("primary").textContent = mode === "replay" ? (playing ? "Pause replay" : "Play replay") :
    mode === "model" ? (automatic ? "Pause after this step" : "Run automatically") : "New run";
  $("step").textContent = mode === "model" ? "One decision" : "Next step";
  $("step").hidden = mode === "manual";
  const ready = state && (mode === "replay" || (config.installed[current] && Number.isInteger(state.revision)));
  $("primary").disabled = !ready || (busy && !automatic) || (mode === "model" && (!link.configured || state.done));
  $("step").disabled = !ready || busy || (mode === "replay" ? index >= replay.steps.length - 1 : (!link.configured || state.done));
  $("reset").disabled = busy;
  $("connect").disabled = !link.editable || busy;
  $("custom-ask").disabled = !link.configured || busy;
  $("scrub").hidden = mode !== "replay";
  $("seed-label").hidden = mode === "replay";
  $("seed").disabled = busy;
  document.querySelectorAll("[data-mode],.case").forEach(button => button.disabled = busy);
  $("actions").querySelectorAll(".action").forEach(button => button.disabled = !ready || mode !== "manual" || busy || state?.done);
}
function updateMetrics(observation) {
  let values;
  if (current === "doom") {
    values = [["Health", Math.round(observation.health ?? 0)], ["Ammo", observation.ammo2 ?? 0],
      ["Kills", observation.killcount ?? 0], ["Armor", observation.armor ?? 0]];
  } else if (current === "crafter") {
    const inventory = observation.inventory || {}, achieved = observation.achievements || {};
    const milestones = ["collect_wood","place_table","make_wood_pickaxe","collect_stone"].filter(key => achieved[key] > 0).length;
    values = [["Health", `${inventory.health ?? 9} / 9`], ["Wood", inventory.wood ?? 0],
      ["Stone", inventory.stone ?? 0], ["Goal milestones", `${milestones} / 4`]];
  } else {
    const checks = observation.checks || {};
    values = [["Peg held", observation.object_between_both_fingers ? "Yes" : "No"],
      ["Gripper", (observation.gripper_open ?? observation.gripper === "open") ? "Open" : "Closed"],
      ["Height above peg", `${Math.round((observation.gripper_height_above_peg_metres ?? observation.gripper_height_above_target_metres ?? 0)*1000)} mm`],
      ["Checks passed", `${Object.values(checks).filter(Boolean).length} / 6`]];
  }
  $("metrics").replaceChildren(...values.map(([name, value]) => {
    const box = element("div", undefined, "metric");
    box.append(element("span", name), element("b", String(value))); return box;
  }));
}
function drawActions(decision) {
  const probabilities = decision?.probabilities;
  $("action-count").textContent = `${Object.keys(actions).length} controls`;
  $("actions-label").textContent = mode === "manual" ? "CHOOSE AN ACTION" : "ACTION CHOICES";
  $("probability-note").textContent = probabilities ? "Original per-option probabilities from this decision." :
    mode === "manual" ? "Each click executes a real environment action." :
    mode === "model" ? "The model's probabilities appear after its first decision." :
    "This preview uses scripted controls. No model probabilities are shown.";
  if (mode === "replay" && replay.controller === "model" && !probabilities)
    $("probability-note").textContent = "Recorded checkpoint probabilities appear at each step.";
  $("actions").replaceChildren(...Object.entries(actions).map(([key, description]) => {
    const button = element("button", undefined, "action" + (decision?.action === key ? " selected" : ""));
    button.dataset.action = key; button.title = description;
    if (probabilities && probabilities[key] !== undefined) {
      const bar = element("span", undefined, "bar"); bar.style.width = `${probabilities[key]*100}%`; button.append(bar);
    }
    const actionName = element("span", label(key), "name");
    if (/^[xyz]_(plus|minus)_(1|5)cm$/.test(key)) actionName.style.textTransform = "none";
    button.append(actionName);
    if (probabilities?.[key] !== undefined) button.append(element("span", `${(probabilities[key]*100).toFixed(1)}%`, "prob"));
    button.onclick = () => liveStep(key);
    return button;
  }));
  const selected = $("actions").querySelector(".selected");
  if (selected) {
    const item = selected.getBoundingClientRect(), list = $("actions").getBoundingClientRect();
    if (item.bottom > list.bottom) $("actions").scrollTop += item.bottom - list.bottom;
    else if (item.top < list.top) $("actions").scrollTop += item.top - list.top;
  }
}
async function show(snapshot, animate = false) {
  state = snapshot;
  if (Object.keys(state.actions || {}).length) actions = state.actions;
  const observation = state.observation || {};
  const frames = state.frames || [];
  const stamp = generation;
  $("step-count").textContent = `STEP ${state.step} / ${mode === "replay" ? replay.steps.length - 1 : config.cases[current].limit}` +
    (observation.image_is_current === false ? " · LAST AVAILABLE FRAME" : "");
  $("step-count").title = observation.image_is_current === false ?
    "The engine returned no image after the episode ended. Counters show the final state." : "";
  $("feedback").textContent = [state.decision?.subgoal, state.feedback].filter(Boolean).join(" ");
  $("feedback-label").textContent = state.done ? (state.success ? "GOAL COMPLETED" : "EPISODE ENDED") : "ENVIRONMENT FEEDBACK";
  $("result-dot").className = state.done ? (state.success ? "success" : "failure") : "";
  $("state-json").textContent = JSON.stringify(observation, null, 2);
  $("scrub").value = index;
  updateMetrics(observation); drawActions(state.decision); controls();
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    for (const frame of frames) {
      if (stamp !== generation) return;
      $("scene").src = frame; await delay(state.frame_duration_ms ?? 65);
    }
  } else if (frames.length) $("scene").src = frames.at(-1);
}
function modeLabels() {
  document.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
  $("provenance").textContent = mode === "replay" ? (replay.controller === "model" ? "RECORDED MODEL RUN" : "SCRIPTED ENVIRONMENT PREVIEW") :
    mode === "manual" ? "LIVE · YOUR CONTROLS" : "LIVE · MODEL DECISIONS";
  $("recording-note").textContent = mode === "replay" ? replay.note :
    "Real environment execution. Simulation time advances only when an action runs." +
    (current === "arm" ? " Motion primitives drive joint motors; contacts and gravity determine the result." : "");
  $("connection").replaceChildren(element("i"), document.createTextNode(mode === "model" && link.model ? link.model : mode === "replay" ? "Local playback" : "Local CPU environment"));
  $("footer-note").textContent = mode === "replay" ? "Playback uses packaged assets only." :
    mode === "manual" ? "Manual play makes no model requests." :
    (link.image_requests ? "The model receives the current image and measured state." : "Text-only: the model receives measured state.");
}
async function selectCase(key) {
  generation++; playing = false; automatic = false; mode = "replay"; current = key; actions = {}; state = null;
  error(""); controls();
  document.querySelectorAll(".case").forEach(button => button.classList.toggle("selected", button.dataset.case === key));
  $("category").textContent = config.cases[key].category;
  $("title").textContent = config.cases[key].title;
  $("goal").textContent = config.cases[key].goal;
  $("scene").className = key === "crafter" ? "pixelated" : "";
  $("scene").alt = config.cases[key].title + " environment view";
  const stamp = generation;
  try {
    const data = await request(`/recordings/${key}/replay.json`);
    if (stamp !== generation) return;
    replay = data; index = 0; $("scrub").max = replay.steps.length - 1;
    $("setup").hidden = true; modeLabels(); await show(replay.steps[0]); startPlayback();
  } catch (e) { error(e.message); }
}
async function startPlayback() {
  if (playing || !replay) return;
  if (index === replay.steps.length - 1) { index = 0; await show(replay.steps[0]); }
  playing = true; controls(); const stamp = generation, playback = ++loop;
  while (playing && playback === loop && mode === "replay" && stamp === generation && index < replay.steps.length - 1) {
    await delay(replay.steps[index + 1].step_pause_ms ?? 950);
    if (!playing || playback !== loop || mode !== "replay" || stamp !== generation) break;
    index++; await show(replay.steps[index], true);
  }
  if (stamp === generation && playback === loop) { playing = false; controls(); }
}
function setupMessage(message, command) {
  $("setup").replaceChildren(element("div", message));
  if (command) $("setup").append(element("code", command));
  $("setup").hidden = false;
  $("provenance").textContent = "REPLAY · LIVE SETUP REQUIRED";
  state = null;
}
async function changeMode(next) {
  generation++; playing = false; automatic = false; mode = next; error(""); modeLabels();
  $("setup").hidden = true;
  if (mode === "replay") { index = 0; await show(replay.steps[0]); return; }
  if (!config.installed[current]) {
    setupMessage("Install the optional CPU environments, then restart the playground. Replay remains available.",
      `python -m pip install -e '.[${config.cases[current].extra}]'`);
    controls(); return;
  }
  if (mode === "model" && !link.configured) {
    setupMessage("Connect a model above: enter the URL of a running server and choose Test and connect. "
      + "Start one with:", "jevany serve --checkpoint runs/my-jev --port 8008");
    controls(); return;
  }
  await newRun();
}
async function newRun() {
  automatic = false; playing = false; generation++; error("");
  busy = true; $("waiting-label").textContent = "Starting the environment"; $("waiting").hidden = false; controls();
  try {
    const snapshot = await request("/api/start", {case: current, seed: Number($("seed").value)});
    actions = snapshot.actions; await show(snapshot);
  } catch (e) { state = null; error(e.message); }
  finally { busy = false; $("waiting").hidden = true; controls(); }
}
async function liveStep(action) {
  if (busy || !state || state.done) return;
  busy = true; error("");
  $("waiting-label").textContent = mode === "model" ? "Waiting for the model" : "Executing your action";
  $("waiting").hidden = false; controls();
  try {
    const body = {revision: state.revision, ...(mode === "model" ? {model: true} : {action})};
    const snapshot = await request("/api/step", body);
    $("waiting").hidden = true; await show(snapshot, true);
    if (snapshot.done) automatic = false;
  } catch (e) { error(e.message); }
  finally { busy = false; $("waiting").hidden = true; controls(); }
}
function renderConnection() {
  const media = link.media || {}, served = link.served;
  $("connect-url").disabled = $("connect-model").disabled = !link.editable;
  if (!$("connect-url").value) $("connect-url").value = link.base_url || link.default_base_url || "";
  if (!$("connect-model").value && link.model && link.model !== "jevany-latest") $("connect-model").value = link.model;
  $("connection-status").textContent = failedAttempt && !link.configured ? "Not connected" :
    !link.configured ? "No model configured" :
    link.reachable ? `Connected: ${link.model}` : `Configured but never tested: ${link.model}`;
  $("connection-status").className = "connect-status " +
    (link.reachable ? "ok" : link.configured ? "warn" : failedAttempt ? "bad" : "");
  $("connection-detail").textContent = served ? [
    `Serves ${served.id}`,
    served.aliases.length ? `aliases ${served.aliases.join(", ")}` : null,
    served.base ? `base ${served.base}` : null,
    served.device ? `device ${served.device}` : null,
    served.decision_mode ? `${served.decision_mode} readout` : null,
    media.model_media_types && media.model_media_types.length
      ? `accepts ${media.model_media_types.join(", ")}` : "text only",
    link.checked ? `checked ${link.checked}` : null,
  ].filter(Boolean).join(" - ") : link.configured
    ? "The identity and media support of this endpoint are unknown until you test it."
    : "Enter the URL of a running JevAny server, then test it. Replays need no model.";
  if (failedAttempt && link.reachable)
    $("connection-detail").textContent += " The last attempt failed; this connected model is still in use.";
  // The box shows what a request would actually carry, not an intent that cannot be met.
  $("images-toggle").checked = !!link.image_requests;
  $("images-toggle").disabled = !link.configured || !link.editable || (!link.image_requests && !media.usable);
  $("images-note").textContent = link.image_requests
    ? `Images are sent using the ${media.transport} transport.`
    : media.usable ? "Text only. Enable images to send the current frame as well."
    : `Text only: ${media.reason || "image input needs a tested model that accepts images"}.`;
  if (!link.configured) $("custom-note").textContent = "Connect a model to ask your own question.";
  else if (!$("custom-probabilities").childElementCount)
    $("custom-note").textContent = "Edit the state, question and options, then ask the model.";
}
function optionRow(option) {
  const row = element("div", undefined, "option-row");
  const name = element("input", undefined, "option-name");
  name.placeholder = "option name"; name.value = option.name || ""; name.spellcheck = false;
  const description = element("input", undefined, "option-description");
  description.placeholder = "what this option means (optional)";
  description.value = option.description || ""; description.spellcheck = false;
  const remove = element("button", "Remove", "option-remove");
  remove.type = "button";
  remove.onclick = () => { if ($("custom-options").children.length > 2) row.remove(); };
  row.append(name, description, remove);
  return row;
}
function customOptions() {
  return [...$("custom-options").children].map(row => ({
    name: row.querySelector(".option-name").value.trim(),
    description: row.querySelector(".option-description").value.trim(),
  })).filter(option => option.name);
}
function drawDecision(result) {
  const best = Math.max(...Object.values(result.probabilities));
  $("custom-answer").textContent = `${result.choice} - confidence ${(result.confidence * 100).toFixed(0)}%`;
  $("custom-probabilities").replaceChildren(...result.options.map(name => {
    const value = result.probabilities[name] ?? 0;
    const row = element("div", undefined, "action" + (name === result.choice ? " selected" : ""));
    const bar = element("span", undefined, "bar"); bar.style.width = `${(value / best) * 100}%`;
    const option = element("span", name, "name"); option.style.textTransform = "none";
    row.append(bar, option, element("span", `${(value * 100).toFixed(1)}%`, "prob"));
    return row;
  }));
  $("custom-note").textContent = `Answered by ${result.model} in ${result.seconds.toFixed(2)} s.`;
}
$("connect").onclick = async () => {
  const url = $("connect-url").value.trim(), model = $("connect-model").value.trim();
  if (!url) { error("Enter the URL of a running model server, for example http://127.0.0.1:8008"); return; }
  busy = true; error(""); $("connect").textContent = "Testing..."; controls();
  try {
    link = await request("/api/connect", model ? {base_url: url, model} : {base_url: url});
    failedAttempt = false;
  } catch (e) {
    failedAttempt = true;
    error("Could not connect: " + e.message);
  }
  finally {
    busy = false; $("connect").textContent = "Test and connect";
    renderConnection(); modeLabels(); controls();
  }
};
$("images-toggle").onchange = async () => {
  const enabled = $("images-toggle").checked;
  error("");
  try { link = await request("/api/images", {enabled}); }
  catch (e) { error(e.message); }
  finally { renderConnection(); modeLabels(); controls(); }
};
$("custom-add").onclick = () => {
  if ($("custom-options").children.length < (config.max_custom_options || 12))
    $("custom-options").append(optionRow({}));
};
$("custom-ask").onclick = async () => {
  const options = customOptions();
  if (options.length < 2) { error("Give at least two named candidate options."); return; }
  busy = true; error(""); $("custom-ask").textContent = "Asking...";
  $("custom-note").textContent = "Waiting for the model."; controls();
  try {
    drawDecision(await request("/api/decide", {
      state: $("custom-state").value, question: $("custom-question").value, options,
    }));
  } catch (e) {
    error(e.message);
    $("custom-note").textContent = "The model did not answer: " + e.message;
  }
  finally { busy = false; $("custom-ask").textContent = "Ask the model"; controls(); }
};
$("primary").onclick = async () => {
  if (mode === "replay") { if (playing) {playing = false; loop++; controls();} else startPlayback(); }
  else if (mode === "manual") await newRun();
  else {
    automatic = !automatic; const run = ++loop; controls();
    while (automatic && run === loop && mode === "model" && state && !state.done) {
      await liveStep(); if (automatic) await delay(Math.min(state.step_pause_ms ?? 250, 250));
    }
    controls();
  }
};
$("step").onclick = async () => {
  playing = false; automatic = false; loop++;
  if (mode === "replay" && index < replay.steps.length - 1) {index++; await show(replay.steps[index]);}
  else if (mode === "model") await liveStep();
};
$("reset").onclick = async () => {
  if (mode === "replay") {generation++; playing = false; index = 0; await show(replay.steps[0]);}
  else await newRun();
};
$("scrub").oninput = async () => {generation++; playing = false; index = Number($("scrub").value); await show(replay.steps[index]);};
document.querySelectorAll("[data-mode]").forEach(button => button.onclick = () => changeMode(button.dataset.mode));
$("download").onclick = async () => {
  try {
    const data = mode === "replay" ? replay : await request("/api/trace");
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:"application/json"}));
    const a = element("a"); a.href = url; a.download = `jevany-${current}-${mode}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { error(e.message); }
};
(async () => {
  try {
    config = await request("/api/config");
    link = config.connection || {};
    $("custom-state").value = EXAMPLE.state;
    $("custom-question").value = EXAMPLE.question;
    $("custom-options").replaceChildren(...EXAMPLE.options.map(optionRow));
    renderConnection();
    for (const [key, meta] of Object.entries(config.cases)) {
      const button = element("button", undefined, "case"); button.dataset.case = key;
      const img = element("img"); img.src = `/recordings/${key}/000.jpg`; img.alt = "";
      const text = element("div"); text.append(element("small", meta.category), element("b", meta.title),
        element("span", key === "crafter" ? "17 native controls" : key === "doom" ? "Freedoom · ViZDoom" : "Franka Panda · PyBullet"));
      button.append(img, text); button.onclick = () => selectCase(key); $("cases").append(button);
    }
    await selectCase(current);
  } catch (e) {error("Could not load the playground: " + e.message);}
})();
