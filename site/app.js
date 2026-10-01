"use strict";

const caseDetails = {
  peg_insertion: {
    title: "robot peg insertion", category: "ROBOTICS",
    description: "Grasp the peg, align it with the socket, and lower it into place.",
  },
  drone: {
    title: "drone delivery", category: "MOBILITY",
    description: "Plan the route, land, and deliver with energy left in reserve.",
  },
  lab: {
    title: "laboratory automation", category: "LABORATORY",
    description: "Calibrate the pipette, transfer the correct doses, and mix without contamination.",
  },
  frontend: {
    title: "frontend repair", category: "BROWSER",
    description: "Repair the checkout layout, then check it at three screen widths.",
  },
  sql: {
    title: "SQL repair", category: "SOFTWARE",
    description: "Repair the revenue query without losing valid orders or counting them twice.",
  },
  chess: {
    title: "tactical chess", category: "TACTICS",
    description: "Choose among legal moves to find a forced mate in two.",
  },
};

const setup = [
  "git clone https://github.com/SimpleJev/JevAny.git",
  "cd JevAny",
  "python3.12 -m venv .venv",
  "source .venv/bin/activate",
];
const recipes = {
  train: {
    commands: [
      "python -m pip install -e '.[train]'",
      "jevany data init --out data/starter",
      "jevany data validate data/starter/train.jsonl",
      "jevany train --config recipes/sft.toml --dry-run",
      "jevany train --config recipes/sft.toml",
    ],
    note: "Python 3.12+ · macOS / Linux shell · This recipe uses Qwen3.5-0.8B on a CUDA GPU with BF16.",
  },
  serve: {
    commands: [
      "python -m pip install -e '.[serve,multimodal]'",
      "jevany serve \\",
      "  --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA \\",
      "  --device cuda --dtype bf16 --port 8008",
    ],
    note: "Python 3.12+ · macOS / Linux shell · CUDA GPU required. Base weights need about 8 GB in BF16, plus runtime memory.",
  },
};

function localRecipe() {
  const model = document.querySelector("#local-model");
  if (model.value === "cpu-starter") {
    return {
      commands: [
        "python -m pip install -e '.[train,serve]'",
        "jevany data init --out data/starter",
        "OMP_NUM_THREADS=4 jevany train --config recipes/sft.toml \\",
        "  --device cpu --dtype fp32 --weights-dtype fp32 \\",
        "  --out runs/cpu-jev",
        "OMP_NUM_THREADS=4 jevany serve --checkpoint runs/cpu-jev \\",
        "  --device cpu --dtype fp32 --port 8008",
      ],
      note: "CPU · 16 GB RAM recommended. Downloads the Qwen 0.8B base and trains a small adapter on the bundled tickets before starting the model.",
    };
  }
  return {
    commands: [
      "python -m pip install -e '.[serve,multimodal]'",
      "jevany serve \\",
      `  --checkpoint ${model.value} \\`,
      "  --device cuda --dtype bf16 --port 8008",
    ],
    note: `CUDA · BF16 base weights need about ${model.selectedOptions[0].dataset.memory} GB, plus runtime memory. Downloads the checkpoint and base on first use; later runs use the local cache.`,
  };
}

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const motionButton = document.querySelector("#motion-toggle");
const replay = document.querySelector("#replay-video");
const replayChoices = document.querySelector(".replay-choices");
const videoExtension = replay.canPlayType('video/webm; codecs="vp9"') ? "webm" : "mp4";
const ambientVideo = document.querySelector(".ambient-video");
const portraitBackground = window.matchMedia("(max-aspect-ratio: 3/4)");
const featuredCases = Object.keys(caseDetails);
let motionEnabled = !reduceMotion.matches && !navigator.connection?.saveData;
let replayVisible = false;
let selectedCase = "peg_insertion";
let manuallyPausedReplay = false;
let completedReplays = 0;
let autoAdvanceReplay = true;
let replayProgressFrame = 0;
let caseOrder = featuredCases;
replay.loop = false;
replayChoices.style.setProperty("--replay-progress", "0");

function play(video) {
  // Autoplay may be denied by the browser; the poster and native controls remain.
  const attempt = video.play();
  if (attempt) attempt.catch(() => {});
}

function canAnimate() {
  return motionEnabled && !document.hidden;
}

function updateReplayProgress() {
  // The ended handler records a completed play before resetting the video time.
  if (autoAdvanceReplay && replay.ended) return;
  const fraction = Number.isFinite(replay.duration) && replay.duration > 0
    ? replay.currentTime / replay.duration : 0;
  const progress = autoAdvanceReplay ? Math.min((completedReplays + fraction) / 2, 1) : 1;
  replayChoices.style.setProperty("--replay-progress", String(progress));
}

function trackReplayProgress() {
  updateReplayProgress();
  if (autoAdvanceReplay && !replay.paused && !replay.ended) {
    replayProgressFrame = requestAnimationFrame(trackReplayProgress);
  }
}

function resumeReplay() {
  if (!canAnimate() || !replayVisible || manuallyPausedReplay) return;
  if (autoAdvanceReplay && completedReplays >= 2) {
    selectCase(featuredCases[(featuredCases.indexOf(selectedCase) + 1) % featuredCases.length]);
    return;
  }
  if (replay.ended) {
    replay.currentTime = 0;
    updateReplayProgress();
  }
  if (replay.paused) play(replay);
}

function syncMotion() {
  motionButton.setAttribute("aria-pressed", String(!motionEnabled));
  motionButton.querySelector(".motion-label").textContent = motionEnabled ? "Pause animation" : "Resume animation";
  motionButton.querySelector(".motion-icon").textContent = motionEnabled ? "Ⅱ" : "▷";
  document.documentElement.classList.toggle("motion-paused", !canAnimate());
  syncBackground();
  if (canAnimate()) {
    if (replayVisible && !manuallyPausedReplay) {
      if (!replay.getAttribute("src")) replay.src = `assets/media/${selectedCase}.${videoExtension}`;
      resumeReplay();
    }
  } else {
    replay.pause();
  }
}

function syncBackground() {
  const layout = portraitBackground.matches ? "grid-mobile" : "grid";
  if (ambientVideo.dataset.layout !== layout) {
    ambientVideo.classList.remove("is-ready");
    if (canAnimate()) {
      ambientVideo.dataset.layout = layout;
      ambientVideo.src = `assets/media/ambient/${layout}.${videoExtension}`;
    }
  }
  if (canAnimate()) {
    if (ambientVideo.paused) play(ambientVideo);
  } else ambientVideo.pause();
}
ambientVideo.addEventListener("loadeddata", () => ambientVideo.classList.add("is-ready"));
ambientVideo.addEventListener("error", () => ambientVideo.classList.remove("is-ready"));
portraitBackground.addEventListener("change", syncBackground);

motionButton.hidden = false;
motionButton.addEventListener("click", () => {
  motionEnabled = !motionEnabled;
  syncMotion();
});
reduceMotion.addEventListener("change", () => {
  motionEnabled = !reduceMotion.matches && !navigator.connection?.saveData;
  syncMotion();
});
document.addEventListener("visibilitychange", syncMotion);

// Three groups keep wide viewports filled through the end of each loop.
const modelTrack = document.querySelector(".model-track");
for (let i = 0; i < 2; i += 1) {
  const duplicateModels = modelTrack.firstElementChild.cloneNode(true);
  duplicateModels.setAttribute("aria-hidden", "true");
  duplicateModels.querySelectorAll("a").forEach(link => { link.tabIndex = -1; });
  modelTrack.append(duplicateModels);
}
document.documentElement.classList.add("motion-ready");
const modelObserver = new IntersectionObserver(entries => {
  document.documentElement.classList.toggle("models-offscreen", !entries[entries.length - 1].isIntersecting);
});
modelObserver.observe(document.querySelector(".model-ribbon"));

// Load only the selected replay, and only when it approaches the viewport.
const observer = new IntersectionObserver(entries => {
  replayVisible = entries[entries.length - 1].isIntersecting;
  if (!replayVisible) replay.pause();
  else {
    // A source is available for explicit playback even when autoplay is off.
    if (!replay.getAttribute("src")) replay.src = `assets/media/${selectedCase}.${videoExtension}`;
    syncMotion();
  }
}, { rootMargin: "100px" });
observer.observe(replay);

function selectCase(name) {
  selectedCase = name;
  completedReplays = 0;
  replayChoices.style.setProperty("--replay-progress", autoAdvanceReplay ? "0" : "1");
  const details = caseDetails[name];
  document.querySelectorAll("[data-case]").forEach(choice => {
    const selected = choice.dataset.case === name;
    choice.classList.toggle("is-active", selected);
    if (choice.tagName === "BUTTON") choice.setAttribute("aria-pressed", String(selected));
  });
  replay.pause();
  replay.poster = `assets/media/${name}.webp`;
  replay.src = `assets/media/${name}.${videoExtension}`;
  replay.setAttribute("aria-label", `Archived JevAny replay: ${details.title}`);
  document.querySelector("#replay-description").textContent = details.description;
  document.querySelector("#replay-number").textContent = `${String(caseOrder.indexOf(name) + 1).padStart(2, "0")} / ${caseOrder.length}`;
  manuallyPausedReplay = false;
  if (canAnimate()) play(replay);
}

document.querySelectorAll("[data-case]").forEach(button => {
  button.addEventListener("click", event => {
    // When the catalog is unavailable, gallery links still open local MP4s.
    if (!caseDetails[button.dataset.case]) return;
    event.preventDefault();
    autoAdvanceReplay = false;
    selectCase(button.dataset.case);
    if (button.classList.contains("case-card")) {
      document.querySelector("#in-action").scrollIntoView();
      replay.focus({ preventScroll: true });
    }
  });
});

// Preserve a pause made with the video's native controls across viewport changes.
replay.addEventListener("pause", () => {
  cancelAnimationFrame(replayProgressFrame);
  updateReplayProgress();
  if (canAnimate() && replayVisible && !replay.ended && !replay.seeking && replay.readyState >= 2) manuallyPausedReplay = true;
});
replay.addEventListener("play", () => {
  manuallyPausedReplay = false;
  cancelAnimationFrame(replayProgressFrame);
  trackReplayProgress();
});
replay.addEventListener("timeupdate", updateReplayProgress);
replay.addEventListener("seeked", updateReplayProgress);
replay.addEventListener("loadedmetadata", updateReplayProgress);
replay.addEventListener("ended", () => {
  if (autoAdvanceReplay) {
    completedReplays += 1;
    replayChoices.style.setProperty("--replay-progress", String(Math.min(completedReplays / 2, 1)));
  }
  resumeReplay();
});

fetch("assets/data/cases.json")
  .then(response => {
    if (!response.ok) throw new Error("Case catalog unavailable");
    return response.json();
  })
  .then(cases => {
    cases.forEach(details => { caseDetails[details.id] = details; });
    caseOrder = cases.map(details => details.id);
    document.querySelector("#replay-number").textContent = `${String(caseOrder.indexOf(selectedCase) + 1).padStart(2, "0")} / ${caseOrder.length}`;
  })
  .catch(() => { /* Gallery links and the six featured selectors remain usable. */ });
syncMotion();

// Open disclosures before following links into the case library or model catalog.
function revealAnchor() {
  const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
  if (!target) return;
  const details = target.closest("details");
  if (details) {
    details.open = true;
    target.scrollIntoView();
  }
}
window.addEventListener("hashchange", revealAnchor);
revealAnchor();
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener("click", () => {
    const target = document.getElementById(link.hash.slice(1));
    if (target?.closest("details")) target.closest("details").open = true;
  });
});

const metricDefinitions = {
  jevbench_public_accuracy: ["JevBench accuracy", "231 public development items", true],
  transfer_v9_accuracy: ["Transfer accuracy", "1,046 clean, knowable decisions", true],
  transfer_v9_nll: ["Negative log-likelihood", "Transfer · Penalizes low probability on the correct answer", false],
  transfer_v9_brier: ["Brier score", "Transfer · Squared error of the predicted probabilities", false],
  transfer_v9_ece: ["Expected calibration error", "Transfer · Gap between confidence and observed accuracy", false],
};
const metricButtons = [...document.querySelectorAll("[data-metric]")];
const chart = document.querySelector(".benchmark-chart");
metricButtons.forEach(button => button.addEventListener("click", () => {
  const key = button.dataset.metric;
  const [title, description, accuracy] = metricDefinitions[key];
  const rows = [...chart.children];
  const value = row => Number(row.getAttribute(`data-${key.replaceAll("_", "-")}`));
  const maximum = accuracy ? 1 : Math.ceil(Math.max(...rows.map(value)) * 10) / 10;
  const format = number => accuracy ? `${(number * 100).toFixed(2)}%` : number.toFixed(3);
  rows.sort((a, b) => accuracy ? value(b) - value(a) : value(a) - value(b));
  rows.forEach(row => {
    row.querySelector(".benchmark-track i").style.width = `${value(row) / maximum * 100}%`;
    row.querySelector(".benchmark-value").textContent = format(value(row));
    chart.append(row);
  });
  metricButtons.forEach(tab => tab.setAttribute("aria-pressed", String(tab === button)));
  document.querySelector("#chart-title").textContent = title;
  document.querySelector("#chart-description").textContent = `${description} · ${accuracy ? "Higher" : "Lower"} is better`;
  document.querySelector(".chart-axis > span:first-child").textContent = accuracy ? "0%" : "0";
  document.querySelector("#chart-axis-end").textContent = accuracy ? "100%" : maximum.toFixed(1);
  document.querySelector("#metric-status").textContent = `${title}. Sorted ${accuracy ? "highest" : "lowest"} first.`;
}));
document.querySelector(".metric-buttons").hidden = false;

const tabs = [...document.querySelectorAll("[data-recipe]")];
function selectRecipe(tab) {
  const local = tab.dataset.recipe === "local";
  const recipe = local ? localRecipe() : recipes[tab.dataset.recipe];
  tabs.forEach(item => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  document.querySelector("#quickstart-panel").setAttribute("aria-labelledby", tab.id);
  document.querySelector("#quickstart-code").textContent = [...setup, ...recipe.commands].join("\n");
  document.querySelector("#recipe-note").textContent = recipe.note;
  document.querySelector("#local-model-control").hidden = !local;
  document.querySelector("#demo-launch").hidden = !local;
  document.querySelector("#command-label").textContent = local ? "1. Start the model" : "Shell commands";
}
document.querySelector("#local-model").addEventListener("change", () => selectRecipe(tabs[0]));
document.querySelectorAll("[data-local-model]").forEach(link => link.addEventListener("click", () => {
  document.querySelector("#local-model").value = link.dataset.localModel;
  selectRecipe(tabs[0]);
}));
selectRecipe(tabs[0]);
tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectRecipe(tab));
  tab.addEventListener("keydown", event => {
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    tabs[next].focus();
    selectRecipe(tabs[next]);
  });
});

document.querySelectorAll("[data-copy-code]").forEach(copyButton => {
  const copyLabel = copyButton.querySelector(".copy-label");
  const code = document.querySelector(copyButton.dataset.copyCode);
  copyButton.hidden = false;
  copyButton.addEventListener("click", async () => {
    const status = document.querySelector("#copy-status");
    try {
      await navigator.clipboard.writeText(code.textContent);
      copyLabel.textContent = "Copied";
      status.textContent = "Commands copied to clipboard.";
      setTimeout(() => { copyLabel.textContent = "Copy commands"; }, 2500);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Clipboard unavailable. Commands selected; press Control+C or Command+C to copy.";
    }
  });
});
