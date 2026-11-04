// Web Audio mixing for one track. The media file carries every stem as a separate channel:
//   0-1 original L/R, 2 enhanced, 3 isolated speech, 4-5 music L/R, 6-7 background L/R.
// Changing a gain changes the mix instantly with no re-processing.

export function gainForStrength(percent) {
  if (percent <= 0) return 0;
  if (percent >= 100) return 1;
  const removal = 1 - percent / 100;
  return Math.pow(10, (-36 * removal * removal) / 20);
}

export function computeGains({ mode, speech, background, music, speechMuted, backgroundMuted, musicMuted, enhanced }) {
  if (!enhanced) return { original: 1, enhanced: 0, isolated: 0, music: 0, background: 0 };
  const e = speech / 100;
  if (mode === "v1") return { original: 1 - e, enhanced: e, isolated: 0, music: 0, background: 0 };
  return {
    original: 0,
    enhanced: speechMuted ? 0 : e,
    isolated: speechMuted ? 0 : 1 - e,
    music: musicMuted ? 0 : gainForStrength(music),
    background: backgroundMuted ? 0 : gainForStrength(background),
  };
}

export class StemPlayer {
  constructor() {
    this.context = null;
    this.element = null;
    this.source = null;
    this.gains = null;
    this.handlers = {};
  }

  // The AudioContext has to wait for a user gesture in most browsers, so it starts on first play.
  _ensureGraph() {
    if (this.gains) return;
    const ctx = (this.context = new AudioContext({ latencyHint: "playback" }));
    const splitter = (this.splitter = ctx.createChannelSplitter(8));
    splitter.channelInterpretation = "discrete";
    splitter.channelCountMode = "explicit";
    const merger = ctx.createChannelMerger(2);
    merger.connect(ctx.destination);
    const gain = () => ctx.createGain();
    const g = { original: [gain(), gain()], enhanced: gain(), isolated: gain(), music: [gain(), gain()], background: [gain(), gain()] };
    this.gains = g;
    const wire = (from, node, side) => {
      splitter.connect(node, from);
      if (side === "both") {
        node.connect(merger, 0, 0);
        node.connect(merger, 0, 1);
      } else {
        node.connect(merger, 0, side);
      }
    };
    wire(0, g.original[0], 0);
    wire(1, g.original[1], 1);
    wire(2, g.enhanced, "both");
    wire(3, g.isolated, "both");
    wire(4, g.music[0], 0);
    wire(5, g.music[1], 1);
    wire(6, g.background[0], 0);
    wire(7, g.background[1], 1);
  }

  _connectElement() {
    if (!this.element || this.source || !this.context) return;
    this.source = this.context.createMediaElementSource(this.element);
    this.source.channelInterpretation = "discrete";
    this.source.connect(this.splitter);
    this.applyGains(this.lastGains, false);
  }

  load(url, isVideo, handlers) {
    this.dispose();
    this.handlers = handlers || {};
    const el = document.createElement(isVideo ? "video" : "audio");
    el.preload = "auto";
    el.crossOrigin = "anonymous";
    el.setAttribute("playsinline", "true");
    el.src = url;
    this.element = el;
    const on = (name, fn) => el.addEventListener(name, fn);
    on("timeupdate", () => this.handlers.onTime?.(el.currentTime));
    on("durationchange", () => this.handlers.onDuration?.(el.duration));
    on("loadedmetadata", () => this.handlers.onDuration?.(el.duration));
    on("progress", () => this.handlers.onBuffered?.(this.bufferedFraction()));
    on("canplay", () => this.handlers.onReady?.(true));
    on("waiting", () => this.handlers.onReady?.(false));
    on("playing", () => {
      this.handlers.onReady?.(true);
      this.handlers.onPlaying?.(true);
    });
    on("pause", () => this.handlers.onPlaying?.(false));
    on("ended", () => this.handlers.onPlaying?.(false));
    on("error", () => this.handlers.onError?.());
    el.load();
    return el;
  }

  bufferedFraction() {
    const el = this.element;
    if (!el || !el.duration || !el.buffered.length) return 0;
    return el.buffered.end(el.buffered.length - 1) / el.duration;
  }

  async toggle() {
    if (!this.element) return;
    if (this.element.paused) await this.play();
    else this.element.pause();
  }

  async play() {
    this._ensureGraph();
    this._connectElement();
    if (this.context.state !== "running") await this.context.resume();
    await this.element.play();
  }

  pause() {
    this.element?.pause();
  }

  seek(seconds) {
    if (this.element) this.element.currentTime = seconds;
  }

  applyGains(values, smooth = true) {
    this.lastGains = values;
    if (!values || !this.gains) return;
    const now = this.context.currentTime;
    const playing = smooth && this.element && !this.element.paused;
    const ramp = (param, value) => {
      if (playing) param.setTargetAtTime(value, now, 0.025);
      else param.setValueAtTime(value, now);
    };
    const g = this.gains;
    g.original.forEach((n) => ramp(n.gain, values.original));
    ramp(g.enhanced.gain, values.enhanced);
    ramp(g.isolated.gain, values.isolated);
    g.music.forEach((n) => ramp(n.gain, values.music));
    g.background.forEach((n) => ramp(n.gain, values.background));
  }

  dispose() {
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    if (this.element) {
      this.element.pause();
      this.element.removeAttribute("src");
      this.element.load();
      this.element = null;
    }
  }
}
