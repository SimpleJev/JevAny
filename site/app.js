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
  preview: {
    commands: ["python -m pip install -e .", "", "jevany demo"],
    note: "Python 3.12+ · macOS / Linux shell · Open localhost:8090 and choose Replay.",
  },
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

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const motionButton = document.querySelector("#motion-toggle");
const replay = document.querySelector("#replay-video");
const videoExtension = replay.canPlayType('video/webm; codecs="vp9"') ? "webm" : "mp4";
const ambientVideos = [...document.querySelectorAll(".ambient-video")];
let motionEnabled = !reduceMotion.matches && !navigator.connection?.saveData;
let replayVisible = false;
let selectedCase = "peg_insertion";
let ambientCases = [];
let ambientIndex = 0;
let activeLayer = 0;
let ambientTimer;
let transitionPending = false;
let manuallyPausedReplay = false;

function play(video) {
  // Autoplay may be denied by the browser; the poster and native controls remain.
  const attempt = video.play();
  if (attempt) attempt.catch(() => {});
}

function canAnimate() {
  return motionEnabled && !document.hidden;
}

function scheduleBackground() {
  clearTimeout(ambientTimer);
  if (canAnimate() && ambientCases.length > 1) {
    ambientTimer = setTimeout(advanceBackground, 16000);
  }
}

function advanceBackground() {
  if (!canAnimate() || transitionPending) return;
  transitionPending = true;
  const nextLayer = 1 - activeLayer;
  const next = ambientVideos[nextLayer];
  ambientIndex = (ambientIndex + 1) % ambientCases.length;
  next.src = `assets/media/ambient/${ambientCases[ambientIndex]}.${videoExtension}`;
  let loadGuard;
  const ready = () => {
    clearTimeout(loadGuard);
    next.removeEventListener("loadeddata", ready);
    next.removeEventListener("error", failed);
    transitionPending = false;
    if (!canAnimate()) return;
    play(next);
    next.classList.add("is-visible");
    ambientVideos[activeLayer].classList.remove("is-visible");
    activeLayer = nextLayer;
    scheduleBackground();
  };
  const failed = () => {
    clearTimeout(loadGuard);
    next.removeEventListener("loadeddata", ready);
    next.removeEventListener("error", failed);
    transitionPending = false;
    scheduleBackground();
  };
  next.addEventListener("loadeddata", ready, { once: true });
  next.addEventListener("error", failed, { once: true });
  loadGuard = setTimeout(failed, 8000);
  next.load();
}

function syncMotion() {
  motionButton.setAttribute("aria-pressed", String(!motionEnabled));
  motionButton.querySelector(".motion-label").textContent = motionEnabled ? "Pause motion" : "Resume motion";
  motionButton.querySelector(".motion-icon").textContent = motionEnabled ? "Ⅱ" : "▷";
  document.documentElement.classList.toggle("motion-paused", !motionEnabled);
  clearTimeout(ambientTimer);
  if (canAnimate()) {
    if (!ambientVideos[activeLayer].getAttribute("src") && ambientCases.length) {
      ambientVideos[activeLayer].src = `assets/media/ambient/${ambientCases[0]}.${videoExtension}`;
      ambientVideos[activeLayer].classList.add("is-visible");
    }
    if (ambientVideos[activeLayer].getAttribute("src")) play(ambientVideos[activeLayer]);
    if (replayVisible && !manuallyPausedReplay) {
      if (!replay.getAttribute("src")) replay.src = `assets/media/${selectedCase}.${videoExtension}`;
      play(replay);
    }
    scheduleBackground();
  } else {
    ambientVideos.forEach(video => video.pause());
    replay.pause();
  }
}

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
ambientVideos.forEach(video => video.addEventListener("transitionend", () => {
  if (!video.classList.contains("is-visible")) video.pause();
}));

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

document.querySelectorAll(".replay-choice").forEach((button, index) => {
  button.addEventListener("click", () => {
    selectedCase = button.dataset.case;
    const details = caseDetails[selectedCase];
    document.querySelectorAll(".replay-choice").forEach(choice => {
      const selected = choice === button;
      choice.classList.toggle("is-active", selected);
      choice.setAttribute("aria-pressed", String(selected));
    });
    replay.pause();
    replay.poster = `assets/media/${selectedCase}.webp`;
    replay.src = `assets/media/${selectedCase}.${videoExtension}`;
    replay.setAttribute("aria-label", `Archived JevAny replay: ${details.title}`);
    document.querySelector("#replay-category").textContent = details.category;
    document.querySelector("#replay-description").textContent = details.description;
    document.querySelector("#replay-number").textContent = `${String(index + 1).padStart(2, "0")} / 06`;
    manuallyPausedReplay = false;
    if (canAnimate()) play(replay);
  });
});

// Preserve a pause made with the video's native controls across viewport changes.
replay.addEventListener("pause", () => {
  if (canAnimate() && replayVisible && !replay.seeking) manuallyPausedReplay = true;
});
replay.addEventListener("play", () => { manuallyPausedReplay = false; });

fetch("assets/media/backgrounds.json")
  .then(response => {
    if (!response.ok) throw new Error("Background catalog unavailable");
    return response.json();
  })
  .then(cases => {
    ambientCases = cases;
    syncMotion();
  })
  .catch(() => { /* Keep the local static background if decorative media fails. */ });
syncMotion();

const tabs = [...document.querySelectorAll("[data-recipe]")];
function selectRecipe(tab) {
  const recipe = recipes[tab.dataset.recipe];
  tabs.forEach(item => {
    const selected = item === tab;
    item.setAttribute("aria-selected", String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  document.querySelector("#quickstart-panel").setAttribute("aria-labelledby", tab.id);
  document.querySelector("#quickstart-code").textContent = [...setup, ...recipe.commands].join("\n");
  document.querySelector("#recipe-note").textContent = recipe.note;
}
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

const copyButton = document.querySelector("#copy-code");
const copyLabel = copyButton.querySelector(".copy-label");
copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(document.querySelector("#quickstart-code").textContent);
    copyLabel.textContent = "Copied";
    status.textContent = "Commands copied to clipboard.";
    setTimeout(() => { copyLabel.textContent = "Copy commands"; }, 2500);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#quickstart-code"));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = "Clipboard unavailable. Commands selected; press Control+C or Command+C to copy.";
  }
});
