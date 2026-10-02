(() => {
  const videos = [...document.querySelectorAll(".harness-video")];
  const overview = document.querySelector("#overview-video");
  if (!overview) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let overviewVisible = false;
  let manualPlayback = false;

  function pauseAll() {
    videos.forEach(video => video.pause());
  }

  function playOverview() {
    if (manualPlayback || reducedMotion.matches || document.hidden || !overviewVisible) return;
    overview.play().catch(() => {});
  }

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00";
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  function addControls(video) {
    const player = document.createElement("div");
    player.className = "player";
    video.before(player);
    player.append(video);

    const controls = document.createElement("div");
    controls.className = "playback-controls";
    controls.setAttribute("role", "group");
    controls.setAttribute("aria-label", `${video.getAttribute("aria-label")} — playback controls`);
    controls.innerHTML = `
      <button type="button" class="playback-toggle">Play</button>
      <input type="range" min="0" max="1" step="0.01" value="0" disabled aria-label="Replay position">
      <span class="playback-time">0:00</span>
      <button type="button" class="playback-expand">Expand</button>`;
    player.append(controls);
    const toggle = controls.querySelector(".playback-toggle");
    const seek = controls.querySelector("input");
    const time = controls.querySelector(".playback-time");
    const expand = controls.querySelector(".playback-expand");

    function update() {
      toggle.textContent = video.paused ? "Play" : "Pause";
      const loaded = Number.isFinite(video.duration) && video.duration > 0;
      seek.disabled = !loaded;
      if (loaded) {
        seek.max = video.duration;
        seek.value = video.currentTime;
      }
      time.textContent = loaded ? `${formatTime(video.currentTime)} / ${formatTime(video.duration)}` : formatTime(video.currentTime);
    }

    controls.addEventListener("pointerdown", () => { manualPlayback = true; });
    controls.addEventListener("keydown", () => { manualPlayback = true; });
    toggle.addEventListener("click", () => {
      manualPlayback = true;
      if (video.paused) video.play().catch(() => {});
      else video.pause();
    });
    seek.addEventListener("input", () => {
      manualPlayback = true;
      video.currentTime = Number(seek.value);
    });
    expand.hidden = !document.fullscreenEnabled;
    expand.addEventListener("click", () => {
      manualPlayback = true;
      const request = document.fullscreenElement ? document.exitFullscreen() : player.requestFullscreen();
      request.catch(() => {});
    });
    document.addEventListener("fullscreenchange", () => {
      expand.textContent = document.fullscreenElement === player ? "Collapse" : "Expand";
    });
    for (const event of ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "seeked"]) {
      video.addEventListener(event, update);
    }
    video.controls = false;
    update();
  }

  for (const video of videos) {
    video.muted = true;
    addControls(video);
    video.addEventListener("pointerdown", () => { manualPlayback = true; });
    video.addEventListener("keydown", () => { manualPlayback = true; });
    video.addEventListener("play", () => {
      if (video !== overview) manualPlayback = true;
      videos.forEach(other => { if (other !== video) other.pause(); });
    });
  }

  const visibility = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.target === overview) {
        overviewVisible = entry.isIntersecting && entry.intersectionRatio >= 0.35;
      }
      if (!entry.isIntersecting || (entry.target === overview && !overviewVisible)) {
        entry.target.pause();
      }
    }
    playOverview();
  }, { threshold: [0, 0.35] });
  videos.forEach(video => visibility.observe(video));

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseAll();
    else playOverview();
  });
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) pauseAll();
    else playOverview();
  });
})();
