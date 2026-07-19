/* ============================================================================
 * ErgoSentinel — Workplace Ergonomic Monitoring
 * Version: 0.4.8-production
 *
 * CHANGELOG v0.4.6 (evaluation freeze — no new features):
 *   Fix 1  index.html: Content-Security-Policy (connect-src 'self') + SRI slots —
 *          egress is now refused by the user agent, not merely absent from source.
 *   Fix 2  scorePosture(): removed the shoulder dead-band double-count. TH was
 *          derived as mean+3SD of neutral self-data, which already exceeds the
 *          1.96SD noise band; subtracting NOISE again pushed the effective trigger
 *          to ~mean+4.9SD and under-fired the feature.
 *   Fix 3  export: monitoredSec — wall time the loop actually ran. sessionDurationSec
 *          includes hidden-tab gaps and OVERSTATES monitoring (observed 64s vs 24s).
 *          monitoredSec is the correct denominator for all exposure ratios (Ch.5).
 *   Fix 4  export: performance.frameTimeMs {p50,p95,p99,max,meanFps} + backendUsed,
 *          evidencing the O3 frame-rate claim and the WebGL configuration.
 *   Fix 5  TH: documented the derivation of strainWarn/strainBad (no behaviour change).
 *   Verified: node --check clean; 43/43 logic assertions passing.
 *
 * CHANGELOG v0.4.2 (field-test correction): real capture data (v0.4.1) yielded
 *   validSampleCount:0 — the geometry gate's absolute 340 px "too-close" ceiling
 *   rejected normal laptop seating. Gate is now RESOLUTION-RELATIVE (fractions of
 *   frame width) and tuned for desk-work proximity, and every rejected frame is
 *   now attributed (gateRejectionReasons in export) so zero-yield sessions are
 *   diagnosable. Re-run a short seated test to confirm validSampleCount > 0.
 *
 * CHANGELOG v0.4.1 (post-remediation patch): resolved 8 architecture/logic gaps —
 *   GAP-01 first-sample window; GAP-02 dead sessionStartMs removed;
 *   GAP-03 detector.dispose() on restart (no WebGL leak); GAP-04 dt clamp;
 *   GAP-05 clamped poor-posture accrual + Page Visibility re-baseline;
 *   GAP-06 post-await running guard; GAP-07 scale-invariant shoulder asymmetry;
 *   GAP-08 validSampleCount + meanValidFrameFraction in export.
 *   Verified: node --check clean, 31/31 logic assertions passing.
 *
 * On-device pose estimation via TensorFlow.js MoveNet (SinglePose Lightning).
 *
 * Pipeline (8 stages):
 *   getUserMedia -> <video> -> MoveNet inference -> 17 keypoints ->
 *   capture-geometry gate -> biomechanical feature extraction ->
 *   REBA-inspired (upper-body, coronal) risk scoring -> time-aware EMA
 *   smoothed strain signal -> JITAI alert engine -> 1 Hz JSON sampling/export.
 *
 * Measurement scope (see dissertation Ch.3 & Ch.5):
 *   A front-on monocular webcam observes the IMAGE (coronal) plane only. It
 *   CANNOT recover sagittal-axis motion, so it does not measure forward-head
 *   posture, lumbar lordosis, or true scapular protraction. This build reports
 *   two coronal constructs — lateral head tilt and shoulder asymmetry — plus a
 *   fail-soft trunk-lean reading available only when the hips are visible.
 *
 * Privacy: zero network egress for video. Inference is WebGL-backed and runs
 * entirely in the browser process. No frame or keypoint data is transmitted.
 *
 * This file consolidates all v0.4 fixes:
 *   BUG-02 export race, BUG-03/04 alert text, BUG-10 reset audit flag,
 *   NEW-01 modal guard+dismiss, NEW-02 full state reset, NEW-03 wall-clock
 *   timestamps, NEW-04 dead-field removal, NEW-05 bilateral AND gate,
 *   Fix-1 frame-rate-decoupled EMA, Fix-2 smoothed-strain rename,
 *   Fix-3 capture-geometry gate, Fix-4/5 valid-frame-fraction sampling,
 *   Fix-7 presence-gated sitting timer, and the MDC dead-band scoring policy.
 * (BUG-01 version string and Fix-8 Subresource Integrity live in index.html.)
 * ========================================================================== */

(() => {
  'use strict';

  // ---------- DOM refs ----------
  const video    = document.getElementById('video');
  const overlay  = document.getElementById('overlay');
  const ctx      = overlay.getContext('2d');
  const loading  = document.getElementById('loading');
  const btnStart = document.getElementById('btn-start');
  const btnStop  = document.getElementById('btn-stop');
  const btnCal   = document.getElementById('btn-calibrate');
  const btnExp   = document.getElementById('btn-export');
  const modal    = document.getElementById('modal');
  const btnBreak = document.getElementById('btn-break-done');

  // v0.4.9: Posture Avatar (UI layer only). Rendered from the same main loop
  // (no second requestAnimationFrame stream) so it cannot desync from the data
  // pipeline or add a second GPU/canvas context race. See UIState below.
  // Defensive lookup: if this HTML is out of sync with this app.js (e.g. the
  // avatar markup wasn't deployed), the whole application must keep working —
  // it must not silently die on a null canvas ref.
  const avatarWrap   = document.getElementById('avatar-wrap');
  const avatarCanvas = document.getElementById('avatar-canvas');
  const avatarCtx    = avatarCanvas ? avatarCanvas.getContext('2d') : null;
  const avatarEnabled = !!(avatarWrap && avatarCanvas && avatarCtx);
  if (!avatarEnabled) {
    console.warn('[ErgoSentinel] Avatar elements (#avatar-wrap / #avatar-canvas) not found in this HTML — '
      + 'avatar rendering disabled, rest of the app is unaffected. Make sure index.html and app.js are the matching v0.4.9 pair.');
  }

  // ---------- State ----------
  const state = {
    detector: null,
    stream: null,
    running: false,
    rafId: null,
    sessionTicker: null,        // setInterval handle for the header clock
    fpsEMA: 0,
    lastT: 0,

    calibration: null,          // reserved for future per-user calibration (still null)

    sessionStartWall: null,     // wall-clock (Date.now) — export timestamps (NEW-03)
    paused: false,              // true while the tab is hidden (Page Visibility, GAP-04/05)

    sittingActiveMs: 0,         // presence-gated sitting accumulator (Fix-7)
    awayMs: 0,                  // time since a valid pose was last present (Fix-7)

    poorPostureMs: 0,
    lastPoorTickMs: null,
    breaksTaken: 0,
    alerts: [],

    smoothedStrain: 0,          // JITAI decision signal (Fix-2; was cumulativeStrain)
    history: [],                // per-second sampled metrics, for export
    lastSampleMs: 0,
    lastAlertKey: {},           // alert throttle map

    calibrationResetCount: 0,   // audit flag for the Reset button (BUG-10)

    windowFramesTotal: 0,       // frames seen in current 1 Hz window (Fix-4/5)
    windowFramesValid: 0,       // frames passing geometry+feature gate (Fix-4/5)
    reasonCounts: {},           // v0.4.2: why frames were rejected (data-quality diagnostics)
    monitoredMs: 0,             // v0.4.6: wall time the loop ACTUALLY ran (excludes hidden-tab gaps)
    frameTimesMs: [],           // v0.4.6: raw inter-frame intervals for performance telemetry
    backendUsed: null,          // v0.4.6: 'webgl' | 'cpu' — evidences the WebGL configuration claim
  };

  // ---------- v0.4.9: Posture Avatar UI state (display only) ----------
  // IMPORTANT: this object is intentionally NOT part of `state`. It is never
  // read by scorePosture/updateSmoothedStrain/maybeSample/exportReport, never
  // reset by resetSessionState(), and never serialised. It exists purely to
  // drive a decorative canvas illustration and carries no research meaning.
  // Angle sign conventions are UI-only (see computeAvatarTargets); the scored/
  // exported features remain the unsigned magnitudes produced by
  // extractFeatures(), unchanged by any of this.
  const UIState = {
    headTiltDeg: 0,      // damped, signed, degrees — for display rotation only
    shoulderTiltDeg: 0,  // damped, signed, degrees
    torsoLeanDeg: 0,     // damped, signed, degrees
    flags: { headTilt: 'good', shoulder: 'good', trunk: 'good' },
    present: false,
  };
  const AVATAR_TAU_SEC = 0.12; // critically-damped response; keeps perceptible
                                // avatar lag well under the 50 ms feedback budget
                                // while smoothing single-frame keypoint jitter.
                                // Uses the SAME time-aware-decay technique as
                                // updateSmoothedStrain() (Fix-1), so avatar
                                // motion is also frame-rate independent.

  function clampNum(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  /**
   * Manual rounded-rect path — does NOT depend on CanvasRenderingContext2D
   * .roundRect(), which is inconsistently available (older Chromium builds,
   * some canvas-fingerprinting-protection extensions, some Incognito privacy
   * hardening strip or shim it). Accepts either a single radius or a
   * [tl, tr, br, bl] array, matching the subset of the native API this file
   * actually uses.
   */
  function avatarRoundRectPath(c, x, y, w, h, radii) {
    const rr = Array.isArray(radii) ? radii : [radii, radii, radii, radii];
    const [tl, tr, br, bl] = rr.map(r => Math.max(0, Math.min(r, Math.min(w, h) / 2)));
    c.moveTo(x + tl, y);
    c.lineTo(x + w - tr, y);
    c.arcTo(x + w, y, x + w, y + tr, tr);
    c.lineTo(x + w, y + h - br);
    c.arcTo(x + w, y + h, x + w - br, y + h, br);
    c.lineTo(x + bl, y + h);
    c.arcTo(x, y + h, x, y + h - bl, bl);
    c.lineTo(x, y + tl);
    c.arcTo(x, y, x + tl, y, tl);
  }

  /**
   * Sign-aware pose targets for the AVATAR ONLY, derived from the same
   * geometry-gated keypoints and the already-computed (unsigned) feature
   * object `f`. This does not modify, re-derive, or duplicate any scored or
   * exported metric — it only recovers the left/right direction that
   * extractFeatures() deliberately discards (direction is not a scored
   * construct there). Magnitudes are taken directly from `f`.
   */
  function computeAvatarTargets(kpts, f) {
    const ls = kpts[KP.leftShoulder], rs = kpts[KP.rightShoulder];
    const shoulderX = (ls.x + rs.x) / 2;

    const shoulderSign = Math.sign(rs.y - ls.y) || 1;
    const shoulderTiltDeg = clampNum(shoulderSign * Math.atan(f.shoulderAsymRatio) * 180 / Math.PI, -10, 10);

    let headTiltDeg = 0;
    const earL = kpts[KP.leftEar], earR = kpts[KP.rightEar];
    const validEarL = earL && earL.score >= TH.minKpConfidence;
    const validEarR = earR && earR.score >= TH.minKpConfidence;
    let headX = null;
    if (validEarL && validEarR) headX = (earL.x + earR.x) / 2;
    else if (kpts[KP.nose] && kpts[KP.nose].score >= TH.minKpConfidence) headX = kpts[KP.nose].x;
    if (headX !== null && f.lateralHeadTiltDeg) {
      const headSign = Math.sign(headX - shoulderX) || 1;
      headTiltDeg = clampNum(headSign * f.lateralHeadTiltDeg, -20, 20); // slider range; saturates visually above 20°
    }

    let torsoLeanDeg = 0;
    const lh = kpts[KP.leftHip], rh = kpts[KP.rightHip];
    if (f.trunkLeanDeg !== null && lh && rh) {
      const hipX = (lh.x + rh.x) / 2;
      const torsoSign = Math.sign(shoulderX - hipX) || 1;
      torsoLeanDeg = clampNum(torsoSign * f.trunkLeanDeg, -15, 15); // slider range; saturates visually above 15°
    }

    return { headTiltDeg, shoulderTiltDeg, torsoLeanDeg };
  }

  /** Time-aware damping toward a target, identical technique to Fix-1's EMA. */
  function dampToward(current, target, dtSec) {
    const alpha = 1 - Math.exp(-Math.max(0, dtSec) / AVATAR_TAU_SEC);
    return current + (target - current) * alpha;
  }

  function updateAvatarState(dtSec, target) {
    if (!avatarEnabled) return;
    UIState.headTiltDeg     = dampToward(UIState.headTiltDeg, target.headTiltDeg, dtSec);
    UIState.shoulderTiltDeg = dampToward(UIState.shoulderTiltDeg, target.shoulderTiltDeg, dtSec);
    UIState.torsoLeanDeg    = dampToward(UIState.torsoLeanDeg, target.torsoLeanDeg, dtSec);
    UIState.flags   = target.flags;
    UIState.present = target.present;
    avatarWrap.classList.toggle('no-pose', !target.present);
  }

  // Palette mirrors the skeleton overlay / pill system exactly (index.html
  // :root custom properties), so the avatar reads as the same visual language
  // rather than a bolted-on component.
  const AVATAR_COLORS = {
    good: '#4ade80', warn: '#facc15', bad: '#f87171', unknown: '#8aa0c2',
    torso: '#16223d', shoulders: '#1f2c4a', skin: '#e6edf7', neck: '#c9d4e8',
  };
  const flagColor = (f) => AVATAR_COLORS[f] || AVATAR_COLORS.unknown;

  // ---------- v0.4.10: Bobblehead sprite assets (CC0, Kenney) ----------
  // Served locally from assets/avatar/ — no network egress, CSP img-src
  // 'self' already permits this with zero policy changes. Preloaded once;
  // drawAvatar() falls back automatically to the existing procedural shapes
  // (drawAvatarProcedural, below, byte-identical to the prior renderer)
  // until every asset is ready, or permanently if any single asset fails to
  // load — same fail-soft philosophy as the hips-not-visible trunk-lean gap
  // elsewhere in this app. This is a pure rendering swap: it does not read
  // or write anything beyond UIState, which itself remains fully separate
  // from the research `state` object (see UIState definition above).
  const AVATAR_ASSET_BASE = 'assets/avatar/';
  const AVATAR_ASSET_FILES = {
    head: 'head.png', face: 'face.png', neck: 'neck.png',
    torso: 'torso.png', arm: 'arm.png', hair: 'hair.png',
  };
  const avatarImages = {};
  let avatarImagesReady = false;

  function loadAvatarAssets() {
    // `Image` does not exist in the Node test harness (test_app.js). This
    // guard means app.js stays requireable there with NO change to any
    // exported/tested behaviour — the avatar simply stays on the procedural
    // renderer, which is exactly what happened before this feature existed.
    if (typeof Image === 'undefined') return;
    const names = Object.keys(AVATAR_ASSET_FILES);
    let remaining = names.length;
    let anyFailed = false;
    names.forEach((name) => {
      const img = new Image();
      img.onload = () => {
        remaining -= 1;
        if (remaining === 0 && !anyFailed) avatarImagesReady = true;
      };
      img.onerror = () => {
        anyFailed = true;
        avatarImagesReady = false;
        console.error(`[ErgoSentinel] Avatar sprite failed to load (${AVATAR_ASSET_FILES[name]}) — `
          + 'falling back to the procedural avatar for this session.');
      };
      img.src = AVATAR_ASSET_BASE + AVATAR_ASSET_FILES[name];
      avatarImages[name] = img;
    });
  }
  loadAvatarAssets();

  /** Small always-on-top status dots — keeps the "clear, high-contrast,
   * skeleton-schema colour" feedback intact regardless of which renderer
   * (sprite or procedural) is active, without tinting artwork we don't own
   * the vector paths for. */
  function drawAvatarFlagDots(c) {
    const items = [
      { label: 'Head',     flag: UIState.flags.headTilt },
      { label: 'Shoulder', flag: UIState.flags.shoulder },
      { label: 'Trunk',    flag: UIState.flags.trunk },
    ];
    c.save();
    c.font = '12px sans-serif';
    c.textBaseline = 'middle';
    items.forEach((it, i) => {
      const y = 16 + i * 20;
      c.fillStyle = flagColor(it.flag);
      c.beginPath(); c.arc(16, y, 5, 0, Math.PI * 2); c.fill();
      c.fillStyle = '#8aa0c2';
      c.fillText(it.label, 28, y);
    });
    c.restore();
  }

  /** Sprite-based renderer. Same 3-tier transform hierarchy as the
   * procedural renderer below (torso -> shoulders/arms -> head), driven by
   * the exact same UIState angles, so live tilt-tracking behaviour is
   * unchanged — only the drawing calls differ. */
  function drawAvatarSprites(c, W, H) {
    const torsoAngle    = UIState.torsoLeanDeg * Math.PI / 180;
    const shoulderAngle = UIState.shoulderTiltDeg * Math.PI / 180;
    const headAngle     = UIState.headTiltDeg * Math.PI / 180;

    // Display-size layout constants, in the canvas's fixed 400x500 native
    // resolution (CSS scales the whole canvas responsively — these do not
    // need to change per screen size). First-pass proportions; easy to
    // retune here if the composited look needs adjusting.
    const torsoW = 202, torsoH = 230;
    const armW = 92, armH = 108;
    const neckW = 58, neckH = 30;
    const headW = 154, headH = 150;
    const faceW = 92, faceH = 93;
    const hairW = 168, hairH = 120;

    c.save();
    c.translate(W / 2, H - 30);

    // Torso tier (torsoLeanDeg)
    c.save();
    c.rotate(torsoAngle);
    c.drawImage(avatarImages.torso, -torsoW / 2, -torsoH, torsoW, torsoH);

    // Shoulder/arm tier (shoulderTiltDeg), pivoting near the collar
    c.translate(0, -torsoH + 46);
    c.rotate(shoulderAngle);
    c.drawImage(avatarImages.arm, -torsoW / 2 - armW * 0.35, -armH * 0.3, armW, armH);
    c.save();
    c.scale(-1, 1);
    c.drawImage(avatarImages.arm, -torsoW / 2 - armW * 0.35, -armH * 0.3, armW, armH);
    c.restore();

    // Head tier (headTiltDeg), pivoting at the neck socket
    c.translate(0, -18);
    c.rotate(headAngle);
    c.drawImage(avatarImages.neck, -neckW / 2, -neckH, neckW, neckH);
    c.drawImage(avatarImages.head, -headW / 2, -headH - neckH + 6, headW, headH);
    c.drawImage(avatarImages.face, -faceW / 2, -headH - neckH + 6 + headH * 0.32, faceW, faceH);
    c.drawImage(avatarImages.hair, -hairW / 2, -headH - neckH - hairH * 0.55, hairW, hairH);

    c.restore(); // shoulder/head rotations
    c.restore(); // origin translate
  }

  /** Original procedural renderer — byte-identical logic to the pre-sprite
   * version, kept as the permanent, dependency-free fallback. */
  function drawAvatarProcedural(c, W, H) {
    const torsoAngle    = UIState.torsoLeanDeg * Math.PI / 180;
    const shoulderAngle = UIState.shoulderTiltDeg * Math.PI / 180;
    const headAngle     = UIState.headTiltDeg * Math.PI / 180;

    const torso = { w: 130, h: 190, r: [14, 14, 60, 60] };
    const shoulders = { w: 190, h: 58, r: 26 };
    const head = { w: 92, h: 108, neckLen: 30, r: 30 };

    c.save();
    c.translate(W / 2, H - 40);

    // Torso
    c.save();
    c.rotate(torsoAngle);
    c.fillStyle = AVATAR_COLORS.torso;
    c.strokeStyle = flagColor(UIState.flags.trunk);
    c.lineWidth = 3;
    c.beginPath();
    avatarRoundRectPath(c, -torso.w / 2, -torso.h, torso.w, torso.h, torso.r);
    c.fill(); c.stroke();

    // Shoulders (relative to torso)
    c.translate(0, -torso.h + 16);
    c.rotate(shoulderAngle);
    c.fillStyle = AVATAR_COLORS.shoulders;
    c.strokeStyle = flagColor(UIState.flags.shoulder);
    c.lineWidth = 3;
    c.beginPath();
    avatarRoundRectPath(c, -shoulders.w / 2, -shoulders.h / 2, shoulders.w, shoulders.h, shoulders.r);
    c.fill(); c.stroke();

    // Head (relative to shoulders)
    c.translate(0, -14);
    c.rotate(headAngle);
    c.fillStyle = AVATAR_COLORS.neck;
    c.fillRect(-16, -head.neckLen, 32, head.neckLen);
    c.fillStyle = AVATAR_COLORS.skin;
    c.strokeStyle = flagColor(UIState.flags.headTilt);
    c.lineWidth = 3;
    c.beginPath();
    avatarRoundRectPath(c, -head.w / 2, -head.h - head.neckLen, head.w, head.h, head.r);
    c.fill(); c.stroke();
    // simple face, for orientation only — not a rendering of any real person
    c.fillStyle = '#33415c';
    c.fillRect(-22, -head.h - head.neckLen + 40, 12, 10);
    c.fillRect(10, -head.h - head.neckLen + 40, 12, 10);
    c.fillRect(-14, -head.h - head.neckLen + 74, 28, 6);
    c.restore(); // head/shoulders/torso rotations
    c.restore(); // origin translate
  }

  /**
   * Renders the avatar from the current (already-damped) UIState. Pure
   * drawing function — reads UIState, writes nothing back to it or to any
   * research state. Called once per main-loop tick (see loop()), never from
   * a second requestAnimationFrame stream. Dispatches to the sprite
   * renderer once assets are ready, otherwise the procedural fallback.
   */
  function drawAvatar() {
    if (!avatarEnabled) return;
    const c = avatarCtx;
    const W = avatarCanvas.width, H = avatarCanvas.height;
    c.clearRect(0, 0, W, H);

    if (avatarImagesReady) {
      drawAvatarSprites(c, W, H);
    } else {
      drawAvatarProcedural(c, W, H);
    }
    drawAvatarFlagDots(c);
  }

  // ---------- MoveNet keypoint indices ----------
  const KP = {
    nose: 0, leftEye: 1, rightEye: 2, leftEar: 3, rightEar: 4,
    leftShoulder: 5, rightShoulder: 6, leftElbow: 7, rightElbow: 8,
    leftWrist: 9, rightWrist: 10, leftHip: 11, rightHip: 12,
    leftKnee: 13, rightKnee: 14, leftAnkle: 15, rightAnkle: 16
  };

  const SKELETON = [
    [KP.leftShoulder, KP.rightShoulder],
    [KP.leftShoulder, KP.leftElbow], [KP.leftElbow, KP.leftWrist],
    [KP.rightShoulder, KP.rightElbow], [KP.rightElbow, KP.rightWrist],
    [KP.leftShoulder, KP.leftHip], [KP.rightShoulder, KP.rightHip],
    [KP.leftHip, KP.rightHip],
    [KP.nose, KP.leftEye], [KP.nose, KP.rightEye],
    [KP.leftEye, KP.leftEar], [KP.rightEye, KP.rightEar],
    [KP.leftShoulder, KP.leftEar], [KP.rightShoulder, KP.rightEar]
  ];

  // ---------- Risk thresholds (research-informed defaults) ----------
  // Demo mode (?demo=1) shortens the micro-break interval for viva demonstration.
  const demoMode = /[?&]demo=1/.test(window.location.search);

  const TH = {
    lateralHeadTiltRatio: 0.25,   // legacy ratio (retained in export for continuity)
    lateralHeadTiltDeg: 15,       // WARN: lateral cervical flexion >=15 deg (REBA/RULA-family, 1 risk pt)
    lateralHeadTiltBadDeg: 30,    // BAD: >30 deg (2 risk pts) — Karhu/observational ergonomics scale
    shoulderAsymRatio: 0.08,      // |shoulder Δy| / shoulder width — CALIBRATED from neutral self-data (mean+3SD; n=1290)
    trunkLeanDeg: 12,             // scored only when hips are visible
    minKpConfidence: 0.30,        // MoveNet keypoint confidence gate
    microBreakIntervalMs: demoMode ? 60 * 1000 : 25 * 60 * 1000,
    sampleIntervalMs: 1000,       // 1 Hz research sampling
    alertThrottleMs: 30 * 1000,   // 30 s alert throttle (JITAI, anti-fatigue)
    strainTauSec: 1.67,           // EMA time constant (Fix-1); matches legacy 30 FPS a=0.02
    awayResetMs: 120 * 1000,      // >=2 min absence => treat as a natural break (Fix-7)
    // Display bands, DERIVED (not arbitrary): the EMA converges on risk*10, so a
    // sustained risk of 3 settles the signal at 30 — and risk >= 3 is exactly the
    // condition tickPoorPosture() treats as poor posture. strainWarn therefore
    // marks "sustained poor posture", and strainBad (60) marks a sustained risk of
    // 6, i.e. roughly two concurrent severe features. These are presentation bands
    // and carry no clinical meaning.
    strainWarn: 30,
    strainBad: 60,
  };

  // Measurement noise floor (dead-band) — MDC principle (dissertation Ch.3/5).
  // The coronal ratio features have no published minimum detectable change, so
  // these MUST be set empirically from a still-neutral self-data session
  // (value = 1.96 * SD(feature) while holding a neutral posture). The clinical
  // exemplar is the ~5 deg MDC reported for the craniovertebral angle
  // (Gallego-Izquierdo et al., 2020). Placeholders below are conservative and
  // must be replaced with measured values before the evaluation study.
  const NOISE = {
    lateralHeadTiltRatio: 0.05,   // legacy ratio (head tilt now scored in degrees)
    shoulderAsymRatio:    0.036,  // CALIBRATED 1.96*SD from neutral self-data (n=1290)
  };

  // Capture-geometry tolerances (Fix-3), expressed as FRACTIONS OF FRAME WIDTH so
  // the gate is resolution-independent and matched to the DSE/desk-work use case
  // (the user is expected to sit CLOSE to a laptop). v0.4.2 recalibration: the old
  // absolute 340 px ceiling (~53% of a 640 px frame) wrongly rejected normal
  // seating as "too-close", yielding zero valid frames in field tests. The upper
  // bound is now deliberately generous; the gate mainly catches "user absent/too
  // far" and gross camera tilt.
  const GEOM = {
    shoulderWidthMinFrac: 0.12,   // below => user too far / partly out of frame
    shoulderWidthMaxFrac: 0.95,   // above => face fills frame (genuinely too close)
    maxRollDeg:           15,     // camera-tilt / trunk-roll tolerance
    minInterocularFrac:   0.05,   // face-landmark ceiling proxy (face features only)
    fallbackFrameWidthPx: 640,    // used if overlay.width is not yet set
  };

  // ---------- Helpers ----------
  const $ = (id) => document.getElementById(id);
  const now = () => performance.now();

  const fmtTime = (ms) => {
    const s = Math.floor(ms / 1000);
    const m = Math.floor(s / 60);
    return m > 0 ? `${m}m ${s % 60}s` : `${s}s`;
  };
  const fmtAngle = (deg) => `${deg.toFixed(1)}\u00B0`;

  /**
   * v0.4.6: summarise raw inter-frame intervals as percentiles.
   * Percentiles (not just a mean) are required because frame-time distributions
   * are right-skewed: a good median can coexist with severe stutter, and only the
   * upper tail reveals it.
   * @param {number[]} arr raw frame times in ms
   * @returns {object|null} {n, p50, p95, p99, max, meanFps} or null when empty
   */
  function frameTimeStats(arr) {
    if (!arr || arr.length === 0) return null;
    const s = arr.slice().sort((a, b) => a - b);
    const at = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    const mean = s.reduce((a, b) => a + b, 0) / s.length;
    return {
      n: s.length,
      p50: +at(0.50).toFixed(2),
      p95: +at(0.95).toFixed(2),
      p99: +at(0.99).toFixed(2),
      max: +s[s.length - 1].toFixed(2),
      meanFps: +(1000 / mean).toFixed(1),
    };
  }

  function pushAlert(level, msg, key) {
    const t = now();
    if (key && state.lastAlertKey[key] && (t - state.lastAlertKey[key]) < TH.alertThrottleMs) return;
    if (key) state.lastAlertKey[key] = t;
    state.alerts.unshift({ t, wallClock: Date.now(), level, msg });
    if (state.alerts.length > 200) state.alerts.pop();
    renderAlerts();
  }

  function renderAlerts() {
    const el = $('alerts');
    if (state.alerts.length === 0) { el.innerHTML = '<div class="alert">No alerts.</div>'; return; }
    el.innerHTML = state.alerts.map(a => {
      const time = new Date(a.wallClock || Date.now()).toLocaleTimeString();
      return `<div class="alert ${a.level}"><div class="t">${time}</div>${a.msg}</div>`;
    }).join('');
  }

  // ---------- Capture-geometry gate (Fix-3) ----------
  /**
   * Assess capture geometry from a single keypoint set.
   * @returns {{ok:boolean, distanceState:string, rollDeg:number,
   *            shoulderWidthPx:number, faceReliable:boolean, reason:string}}
   */
  function assessCaptureGeometry(kpts) {
    const v = (i) => kpts[i] && kpts[i].score >= TH.minKpConfidence;
    if (!v(KP.leftShoulder) || !v(KP.rightShoulder)) {
      return { ok: false, distanceState: 'unknown', rollDeg: 0,
               shoulderWidthPx: 0, faceReliable: false, reason: 'shoulders-not-visible' };
    }
    const ls = kpts[KP.leftShoulder], rs = kpts[KP.rightShoulder];
    const dx = rs.x - ls.x, dy = rs.y - ls.y;
    const shoulderWidthPx = Math.hypot(dx, dy);
    // Roll = deviation of the shoulder line from horizontal, INDEPENDENT of which
    // shoulder keypoint has the larger image-x. v0.4.3 fix: the previous
    // atan2(dy, dx) returned ~180 deg (not ~0) whenever MoveNet's leftShoulder
    // sat to the right of rightShoulder in image space — routine on a front
    // camera due to anatomical labelling vs. the mirrored preview — so level
    // shoulders were mis-read as fully tilted and every frame was rejected as
    // 'camera-tilt'. Using |dy| / |dx| removes the left/right ordering entirely.
    const rollDeg = Math.abs(Math.atan2(Math.abs(dy), Math.abs(dx)) * 180 / Math.PI);

    // Resolution-relative distance band (fraction of frame width), so the same
    // tolerances hold across camera resolutions and framings (GAP: v0.4.2 fix).
    const frameW = (overlay && overlay.width) ? overlay.width : GEOM.fallbackFrameWidthPx;
    const minPx = GEOM.shoulderWidthMinFrac * frameW;
    const maxPx = GEOM.shoulderWidthMaxFrac * frameW;

    let distanceState = 'ok';
    if (shoulderWidthPx < minPx)      distanceState = 'too-far';
    else if (shoulderWidthPx > maxPx) distanceState = 'too-close';

    // Face-landmark reliability ceiling via inter-ocular span (face features only).
    let faceReliable = false;
    if (v(KP.leftEye) && v(KP.rightEye)) {
      const iod = Math.hypot(kpts[KP.rightEye].x - kpts[KP.leftEye].x,
                             kpts[KP.rightEye].y - kpts[KP.leftEye].y);
      faceReliable = iod >= GEOM.minInterocularFrac * frameW;
    }

    const ok = (distanceState === 'ok') && (rollDeg <= GEOM.maxRollDeg);
    const reason = ok ? 'ok'
                 : (distanceState !== 'ok' ? distanceState : 'camera-tilt');
    return { ok, distanceState, rollDeg, shoulderWidthPx, faceReliable, reason };
  }

  // ---------- Pose feature extraction ----------
  /**
   * Extract coronal-plane features. Returns null if the shoulders (the anchor
   * for every feature) are not reliably visible.
   */
  function extractFeatures(kpts) {
    const valid = (i) => kpts[i] && kpts[i].score >= TH.minKpConfidence;
    if (!valid(KP.leftShoulder) || !valid(KP.rightShoulder)) return null;

    const ls = kpts[KP.leftShoulder], rs = kpts[KP.rightShoulder];
    // GAP-07: single scale reference (Euclidean shoulder width), shared with the
    // geometry gate. Roll is gated <=12 deg upstream, so this ~= |dx| in practice.
    const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
    if (shoulderWidth < 20) return null; // person too small / off camera
    const shoulderX = (ls.x + rs.x) / 2;

    // Lateral head tilt — bilateral AND gate with nose fallback (NEW-05).
    // NOTE: this is lateral (coronal) inclination, NOT sagittal forward-head
    // posture, which a front-on monocular camera cannot observe.
    // v0.4.4: also compute the tilt as an ANGLE from vertical (degrees), so the
    // alert threshold can use the research-backed lateral cervical-flexion cutoffs
    // (>=15 deg = 1 risk pt, >30 deg = 2 pts) rather than an un-anchored ratio.
    let lateralHeadTiltRatio = 0;
    let lateralHeadTiltDeg = 0;
    let headX = null, headY = null;
    const earL = valid(KP.leftEar) ? kpts[KP.leftEar] : null;
    const earR = valid(KP.rightEar) ? kpts[KP.rightEar] : null;
    if (earL && earR) {
      headX = (earL.x + earR.x) / 2;                         // symmetric midpoint
      headY = (earL.y + earR.y) / 2;
      lateralHeadTiltRatio = Math.abs(headX - shoulderX) / shoulderWidth;
    } else if (valid(KP.nose)) {
      // Nose is anatomically symmetric; 0.7 compensates for the shorter baseline.
      headX = kpts[KP.nose].x;
      headY = kpts[KP.nose].y;
      lateralHeadTiltRatio = Math.abs(headX - shoulderX) / shoulderWidth * 0.7;
    }
    if (headX !== null) {
      // Angle of the head-to-shoulder-midpoint line from vertical (coronal plane).
      // Capped at 45 deg: this is the top of the lateral cervical-flexion risk band
      // (>45 deg = max risk), and it avoids the atan2 nonlinearity as the head nears
      // shoulder height (vertical distance -> 0 drives the raw angle toward 90 deg,
      // which over-states a hard-but-finite lean).
      const shoulderMidY = (ls.y + rs.y) / 2;
      const horiz = Math.abs(headX - shoulderX);
      const vert  = Math.abs(headY - shoulderMidY);
      const raw   = (vert > 1e-3) ? Math.atan2(horiz, vert) * 180 / Math.PI : 45;
      lateralHeadTiltDeg = Math.min(raw, 45);
    }
    // else: no reliable head landmark -> stays 0 (no false alert)

    // Shoulder asymmetry (frontal-plane tilt), reported as a coronal proxy.
    // GAP-07: normalised by SHOULDER WIDTH (scale/resolution invariant), not by
    // frame height. This makes the metric comparable across cameras and framing.
    // NOTE: threshold + noise floor for this feature are in shoulder-width units
    // now (a slope), so TH.shoulderAsymRatio and NOISE.shoulderAsymRatio MUST be
    // recalibrated from still-neutral self-data before the evaluation study.
    const shoulderAsymRatio = Math.abs(ls.y - rs.y) / shoulderWidth;

    // Trunk lean: null when hips are not visible (frequently cropped by the desk).
    let trunkLeanDeg = null;
    const hipsVisible = valid(KP.leftHip) && valid(KP.rightHip);
    if (hipsVisible) {
      const sm = { x: shoulderX, y: (ls.y + rs.y) / 2 };
      const hm = { x: (kpts[KP.leftHip].x + kpts[KP.rightHip].x) / 2,
                   y: (kpts[KP.leftHip].y + kpts[KP.rightHip].y) / 2 };
      const dx = sm.x - hm.x;
      const dy = sm.y - hm.y; // negative when shoulders sit above hips, as expected
      trunkLeanDeg = Math.abs(Math.atan2(dx, -dy)) * (180 / Math.PI);
    }

    // v0.4.8 (OI-2): compute detection confidence over the keypoints the coronal
    // features ACTUALLY use. The previous set averaged in both hips, which are
    // occluded by the desk in the overwhelming majority of seated frames; their
    // near-zero scores dragged the reported figure down by roughly 2/7 (a display
    // reading of ~44% on an otherwise healthy detection). Hip visibility is
    // reported separately and honestly via the `hipsVisible` sample field, which
    // is where that information belongs. Display-only: not used in scored risk.
    const measured = [KP.nose, KP.leftEar, KP.rightEar, KP.leftShoulder, KP.rightShoulder];
    const meanConf = measured.reduce((s, i) => s + (kpts[i]?.score || 0), 0) / measured.length;

    return { lateralHeadTiltRatio, lateralHeadTiltDeg, shoulderAsymRatio, trunkLeanDeg, shoulderWidth, meanConf };
  }

  // ---------- Risk scoring (REBA-inspired, upper-body, MDC dead-band) ----------
  function scorePosture(f) {
    let risk = 0;
    const flags = { headTilt: 'good', shoulder: 'good', trunk: 'good' };

    // Lateral head tilt — scored on the ANGLE (degrees) against the research-backed
    // lateral cervical-flexion cutoffs (>=15 deg warn, >30 deg bad). The 15 deg
    // threshold sits an order of magnitude above the ~2 deg neutral-sitting noise
    // observed in self-data, so it is inherently robust to measurement jitter
    // (MDC principle) without an arbitrary ratio dead-band.
    if (f.lateralHeadTiltDeg > TH.lateralHeadTiltBadDeg) { risk += 4; flags.headTilt = 'bad'; }
    else if (f.lateralHeadTiltDeg > TH.lateralHeadTiltDeg) { risk += 2; flags.headTilt = 'warn'; }

    // Shoulder asymmetry — dead-banded.
    // v0.4.6: compare DIRECTLY against the calibrated threshold — no dead-band
    // subtraction. TH.shoulderAsymRatio was derived as mean + 3*SD of neutral
    // self-data (n=1290), which by construction already exceeds the 1.96*SD (95%)
    // measurement band recorded in NOISE. Subtracting NOISE first (as v0.4.5 did)
    // double-counted measurement error, pushing the effective trigger to ~mean+4.9SD
    // and under-firing the feature. This also harmonises the treatment of the two
    // coronal features: lateral head tilt is likewise compared directly (its 15 deg
    // band sits an order of magnitude above the 2.37 deg neutral noise band).
    // NOISE is retained in the export as measurement-provenance metadata.
    if (f.shoulderAsymRatio > TH.shoulderAsymRatio * 2) { risk += 3; flags.shoulder = 'bad'; }
    else if (f.shoulderAsymRatio > TH.shoulderAsymRatio) { risk += 1; flags.shoulder = 'warn'; }

    // Trunk lean — scored only when hips are visible.
    if (f.trunkLeanDeg !== null) {
      if (f.trunkLeanDeg > TH.trunkLeanDeg * 1.8) { risk += 4; flags.trunk = 'bad'; }
      else if (f.trunkLeanDeg > TH.trunkLeanDeg)  { risk += 2; flags.trunk = 'warn'; }
    } else {
      flags.trunk = 'unknown'; // honest signal that the feature cannot be measured
    }

    return { risk, flags };
  }

  // ---------- Time-aware smoothed strain signal (Fix-1) ----------
  /**
   * EMA update decoupled from frame rate: alpha_dt = 1 - exp(-dt/tau).
   * @param {number} riskScaled  scored.risk * 10, mapped into the 0..100 domain
   * @param {number} dtSec       elapsed seconds since the previous update
   * @returns {number} clamped smoothed strain (0..100)
   */
  function updateSmoothedStrain(riskScaled, dtSec) {
    const dt = Math.max(0, dtSec);
    const alpha = 1 - Math.exp(-dt / TH.strainTauSec);
    state.smoothedStrain = (1 - alpha) * state.smoothedStrain + alpha * riskScaled;
    if (state.smoothedStrain > 100) state.smoothedStrain = 100;
    if (state.smoothedStrain < 0) state.smoothedStrain = 0;
    return state.smoothedStrain;
  }

  // ---------- Presence-gated sitting timer (Fix-7) ----------
  /**
   * Accrue sitting time only while a valid pose is present. A prolonged
   * absence (TH.awayResetMs) is treated as a natural break and resets the clock.
   */
  function tickPresence(posePresent, dtMs) {
    const dt = Math.max(0, dtMs);
    if (posePresent) {
      state.sittingActiveMs += dt;
      state.awayMs = 0;
    } else {
      state.awayMs += dt;
      if (state.awayMs >= TH.awayResetMs) {
        state.sittingActiveMs = 0;
        state.awayMs = 0;
      }
    }
  }

  // ---------- Drawing ----------
  function drawSkeleton(kpts) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.lineWidth = 3;
    SKELETON.forEach(([a, b]) => {
      const ka = kpts[a], kb = kpts[b];
      if (!ka || !kb || ka.score < TH.minKpConfidence || kb.score < TH.minKpConfidence) return;
      ctx.strokeStyle = '#4ade80';
      ctx.beginPath();
      ctx.moveTo(ka.x, ka.y);
      ctx.lineTo(kb.x, kb.y);
      ctx.stroke();
    });
    kpts.forEach(k => {
      if (!k || k.score < TH.minKpConfidence) return;
      ctx.fillStyle = '#e6edf7';
      ctx.beginPath();
      ctx.arc(k.x, k.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function setPill(id, label, level) {
    const el = $(id);
    el.textContent = label;
    el.className = 'pill ' + (level === 'good' ? 'good' : level === 'warn' ? 'warn' : level === 'bad' ? 'bad' : '');
  }

  function updateMetrics(f, scored) {
    if (!f) return;
    $('m-neck').textContent = `${f.lateralHeadTiltDeg.toFixed(0)}\u00B0 tilt`;
    $('m-neck').className = 'v ' + (scored.flags.headTilt === 'bad' ? 'bad' : scored.flags.headTilt === 'warn' ? 'warn' : '');

    $('m-shoulder').textContent = `${(f.shoulderAsymRatio * 100).toFixed(1)}%`;
    $('m-shoulder').className = 'v ' + (scored.flags.shoulder === 'bad' ? 'bad' : scored.flags.shoulder === 'warn' ? 'warn' : '');

    $('m-trunk').textContent = (f.trunkLeanDeg !== null) ? fmtAngle(f.trunkLeanDeg) : 'hips not visible';
    $('m-trunk').className = 'v ' + (scored.flags.trunk === 'bad' ? 'bad' : scored.flags.trunk === 'warn' ? 'warn' : '');

    $('m-poor-time').textContent = fmtTime(state.poorPostureMs);
    $('m-sit-time').textContent  = fmtTime(state.sittingActiveMs);
    $('m-breaks').textContent    = String(state.breaksTaken);

    // Smoothed strain signal is updated in the loop (Fix-1); here we only display.
    const s = Math.min(100, Math.round(state.smoothedStrain));
    const num = $('score-num');
    num.textContent = s;
    num.className = 'num ' + (s > TH.strainBad ? 'bad' : s > TH.strainWarn ? 'warn' : '');

    setPill('pill-neck',     `Head tilt: ${scored.flags.headTilt.toUpperCase()}`, scored.flags.headTilt);
    setPill('pill-shoulder', `Shoulders: ${scored.flags.shoulder.toUpperCase()}`, scored.flags.shoulder);
    const trunkLabel = scored.flags.trunk === 'unknown' ? 'Trunk: N/A (hips hidden)' : `Trunk: ${scored.flags.trunk.toUpperCase()}`;
    setPill('pill-trunk', trunkLabel, scored.flags.trunk === 'unknown' ? 'warn' : scored.flags.trunk);
    setPill('pill-conf', `Detection: ${(f.meanConf * 100).toFixed(0)}%`,
            f.meanConf > 0.6 ? 'good' : f.meanConf > 0.4 ? 'warn' : 'bad');
  }

  function alertOnFlags(scored) {
    // BUG-03/04: alert text accurately describes lateral (coronal) tilt only.
    if (scored.flags.headTilt === 'bad') {
      pushAlert('bad',  'Strong lateral head tilt detected. Check that your monitors and seating are symmetric.', 'head-bad');
    } else if (scored.flags.headTilt === 'warn') {
      pushAlert('warn', 'Lateral head tilt detected \u2014 head leaning to one side. Check monitor and chair height symmetry.', 'head-warn');
    }
    if (scored.flags.shoulder === 'bad') {
      pushAlert('bad',  'Strong shoulder asymmetry detected. Check chair armrest height and desk symmetry.', 'sh-bad');
    }
    if (scored.flags.trunk === 'bad') {
      pushAlert('bad',  'Significant trunk lean. Realign hips against the backrest.', 'trunk-bad');
    }
  }

  // ---------- Sampling + persistence (Fix-4/5) ----------
  function maybeSample(f, scored) {
    if (now() - state.lastSampleMs < TH.sampleIntervalMs) return;
    state.lastSampleMs = now();

    const total = Math.max(1, state.windowFramesTotal);
    const validFrameFraction = +(state.windowFramesValid / total).toFixed(3);

    state.history.push({
      t: Date.now(),
      lateralHeadTiltRatio: f ? +f.lateralHeadTiltRatio.toFixed(3) : null,
      lateralHeadTiltDeg:   (f && f.lateralHeadTiltDeg != null) ? +f.lateralHeadTiltDeg.toFixed(1) : null,
      shoulderAsymRatio:    f ? +f.shoulderAsymRatio.toFixed(3)    : null,
      trunkLeanDeg: (f && f.trunkLeanDeg !== null) ? +f.trunkLeanDeg.toFixed(1) : null,
      hipsVisible:  !!(f && f.trunkLeanDeg !== null),
      risk:   scored ? scored.risk : null,
      strain: Math.round(state.smoothedStrain),
      validFrameFraction,                 // data-quality metadata (Ch.5)
      framesInWindow: state.windowFramesTotal,
    });

    state.windowFramesTotal = 0;
    state.windowFramesValid = 0;

    // BUG-02: export becomes valid only once at least one sample exists.
    if (state.history.length === 1) btnExp.disabled = false;
  }

  function tickPoorPosture(scored) {
    const isPoor = scored.risk >= 3;
    const t = now();
    if (isPoor) {
      if (state.lastPoorTickMs) {
        // GAP-05: never absorb a render stall / backgrounding gap into cumulative
        // exposure. A single frame can represent at most one sampling period.
        const inc = Math.min(t - state.lastPoorTickMs, TH.sampleIntervalMs);
        state.poorPostureMs += inc;
      }
      state.lastPoorTickMs = t;
    } else {
      state.lastPoorTickMs = null;
    }
  }

  function maybeMicroBreak() {
    if (modal.classList.contains('show')) return;               // NEW-01 idempotent guard
    if (state.sittingActiveMs >= TH.microBreakIntervalMs) {     // Fix-7 accumulator
      modal.classList.add('show');
    }
  }

  // ---------- Main loop ----------
  async function loop() {
    if (!state.running) return;

    const t     = now();
    const dtRaw = t - state.lastT;         // true ms since previous frame (for FPS)
    // GAP-04: clamp the integration timestep so a render stall or a
    // background-tab resume cannot inject a giant dt into the strain EMA or the
    // sitting/away timers. One frame integrates at most one sampling period.
    const dt = Math.min(Math.max(0, dtRaw), TH.sampleIntervalMs);
    state.lastT = t;

    // v0.4.6 (Fix 3): accumulate the time the loop ACTUALLY ran. The clamped dt is
    // used deliberately, so a hidden-tab gap (rAF is suspended by the browser) is
    // NOT counted as monitored time. This is the correct denominator for every
    // exposure ratio in the evaluation; sessionDurationSec is wall-clock and
    // overstates monitoring whenever the tab was backgrounded.
    state.monitoredMs += dt;

    // v0.4.6 (Fix 4): retain raw inter-frame intervals for performance telemetry.
    // Bounded to protect memory on long sessions (8 h at 60 fps would be ~1.7M).
    if (dtRaw > 0 && dtRaw < 10000 && state.frameTimesMs.length < 200000) {
      state.frameTimesMs.push(dtRaw);
    }

    state.fpsEMA = state.fpsEMA
      ? 0.9 * state.fpsEMA + 0.1 * (1000 / Math.max(1, dtRaw))
      : (1000 / Math.max(1, dtRaw));
    setPill('pill-fps', `FPS: ${state.fpsEMA.toFixed(0)}`,
            state.fpsEMA > 15 ? 'good' : state.fpsEMA > 8 ? 'warn' : 'bad');

    state.windowFramesTotal += 1;          // Fix-4/5: count every frame
    let posePresent = false;

    // v0.4.9: default avatar target — neutral pose, dims via UIState.present.
    // Overwritten below only when a valid feature set exists; never affects
    // any research/data variable in this function.
    let avatarTarget = {
      headTiltDeg: 0, shoulderTiltDeg: 0, torsoLeanDeg: 0,
      flags: { headTilt: 'good', shoulder: 'good', trunk: 'good' },
      present: false,
    };

    try {
      const poses = await state.detector.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
      if (!state.running) return;          // GAP-06: stop() happened during inference
      if (poses.length > 0) {
        const kpts = poses[0].keypoints;
        drawSkeleton(kpts);

        const geom = assessCaptureGeometry(kpts);   // Fix-3
        posePresent = geom.ok;

        if (geom.ok) {
          const f = extractFeatures(kpts);
          if (f) {
            const scored = scorePosture(f);
            updateSmoothedStrain(scored.risk * 10, dt / 1000);  // Fix-1: time-aware
            state.windowFramesValid += 1;                       // Fix-4/5: valid frame
            updateMetrics(f, scored);
            tickPoorPosture(scored);
            alertOnFlags(scored);
            maybeSample(f, scored);
            // v0.4.9: avatar reads the SAME f/scored objects, read-only.
            // Isolated try/catch: an avatar bug must be diagnosable on its own
            // and must never be mislabeled as (or hide behind) an inference
            // error, and must never affect any of the scoring/export calls
            // above, which have already completed by this point.
            try {
              avatarTarget = { ...computeAvatarTargets(kpts, f), flags: scored.flags, present: true };
            } catch (avErr) {
              console.error('[ErgoSentinel] Avatar target computation error:', avErr);
            }
          } else {
            state.reasonCounts['features-null'] = (state.reasonCounts['features-null'] || 0) + 1;
            maybeSample(null, null);
          }
        } else {
          state.reasonCounts[geom.reason] = (state.reasonCounts[geom.reason] || 0) + 1;  // v0.4.2 diagnostics
          setPill('pill-conf', `Adjust camera: ${geom.reason}`, 'warn'); // user hint
          maybeSample(null, null);                  // record the gap honestly
        }
      } else {
        state.reasonCounts['no-pose'] = (state.reasonCounts['no-pose'] || 0) + 1;  // v0.4.2 diagnostics
        maybeSample(null, null);
      }
    } catch (err) {
      console.error('Inference error:', err);
    }

    tickPresence(posePresent, dt);        // Fix-7: presence-gated accrual
    maybeMicroBreak();

    // v0.4.9: avatar update+draw, synchronised to this SAME tick (no second
    // rAF stream, no extra GPU/canvas context switching). Uses the same
    // clamped dt as the research timers, so a backgrounded-tab resume cannot
    // snap the avatar any more than it can corrupt the strain EMA. Isolated
    // try/catch so an avatar rendering fault can NEVER stop the main loop
    // from rescheduling (the line below this block is load-bearing for the
    // whole app, not just the avatar).
    try {
      updateAvatarState(dt / 1000, avatarTarget);
      drawAvatar();
    } catch (avErr) {
      console.error('[ErgoSentinel] Avatar render error:', avErr);
    }

    state.rafId = requestAnimationFrame(loop);
  }

  // ---------- Lifecycle ----------
  function resetSessionState() {
    // NEW-02: every session begins from a fully clean state.
    state.history = [];
    state.smoothedStrain = 0;
    state.poorPostureMs = 0;
    state.lastPoorTickMs = null;
    state.breaksTaken = 0;
    state.alerts = [];
    state.lastAlertKey = {};
    state.fpsEMA = 0;
    state.lastSampleMs = 0;
    state.sittingActiveMs = 0;
    state.awayMs = 0;
    state.windowFramesTotal = 0;
    state.windowFramesValid = 0;
    state.reasonCounts = {};
    state.monitoredMs = 0;        // v0.4.6
    state.frameTimesMs = [];      // v0.4.6
    state.calibrationResetCount = 0;
  }

  async function startSession() {
    btnStart.disabled = true;
    btnExp.disabled = true;                // BUG-02: no export before first sample
    loading.textContent = 'Loading MoveNet model (\u22483 MB)\u2026';

    // ---- STAGE 1: model load (network/CSP failures surface HERE, not as camera errors) ----
    // v0.4.7: previously a single try/catch wrapped model load AND camera acquisition,
    // so a blocked model download reported "Check camera permissions" — misleading.
    try {
      await tf.ready();
      try {
        await tf.setBackend('webgl');
        state.backendUsed = 'webgl';        // v0.4.6: evidence for the O3 performance claim
        console.log('[ErgoSentinel] WebGL backend active.');
      } catch (e) {
        console.warn('[ErgoSentinel] WebGL unavailable, falling back to CPU:', e);
        await tf.setBackend('cpu');
        state.backendUsed = 'cpu';          // v0.4.6
        pushAlert('warn', 'WebGL is not available on this browser/hardware. Running on CPU \u2014 frame rate will be reduced.', 'webgl-fallback');
      }
      // GAP-03: TF.js does not garbage-collect tensors; dispose the previous
      // detector before building a new one to avoid a WebGL memory leak across
      // start/stop/start cycles.
      if (state.detector && typeof state.detector.dispose === 'function') {
        state.detector.dispose();
        state.detector = null;
      }
      state.detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
      );
    } catch (err) {
      console.error('[ErgoSentinel] Model load failed:', err);
      const networkish = /fetch|network|load|blocked|refused|csp|content security/i.test(String(err && err.message));
      loading.style.display = 'block';
      loading.textContent = networkish
        ? 'Could not download the MoveNet model. This is a NETWORK or CONTENT-SECURITY-POLICY problem \u2014 NOT a camera problem. '
          + 'Open DevTools (F12) \u2192 Console. If a Content-Security-Policy violation is listed, add the blocked host to connect-src in index.html, '
          + 'or bundle the model locally (see README). Original error: ' + err.message
        : 'Failed to initialise the pose model: ' + err.message;
      btnStart.disabled = false;
      return;
    }

    // ---- STAGE 2: camera acquisition (genuine permission errors surface HERE) ----
    try {
      loading.textContent = 'Requesting camera permission\u2026';
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
      });
      video.srcObject = state.stream;
      await new Promise((res, rej) => {
        const timeout = setTimeout(() => rej(new Error('Video metadata load timed out after 5 seconds.')), 5000);
        video.addEventListener('loadedmetadata', () => { clearTimeout(timeout); res(); }, { once: true });
      });
      await video.play();
    } catch (err) {
      console.error('[ErgoSentinel] Camera acquisition failed:', err);
      loading.style.display = 'block';
      loading.textContent = 'Camera unavailable: ' + err.message
        + '. Check the camera permission icon in the browser address bar, and that no other application is using the webcam.';
      btnStart.disabled = false;
      return;
    }

    // ---- STAGE 3: session start ----
    try {

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      loading.style.display = 'none';

      // ---- clean state, then start clocks (NEW-02 / NEW-03) ----
      resetSessionState();
      state.sessionStartWall = Date.now();   // wall-clock reference — no drift
      state.running = true;
      state.paused = false;
      state.lastT = now();
      state.lastSampleMs = now();            // GAP-01: first sample = a real 1 s window

      pushAlert('good', 'Session started. The system will warn you if it detects sustained poor posture.', 'start');
      if (demoMode) {
        pushAlert('warn', 'Demo mode active \u2014 micro-break interval shortened to 60 seconds for demonstration.', 'demo-mode');
      }

      btnStop.disabled = false;
      btnCal.disabled = false;
      // btnExp stays disabled until the first sample is recorded (BUG-02).

      const sessionStart = state.sessionStartWall;
      state.sessionTicker = setInterval(() => {
        if (!state.running) return;
        const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
        $('session-info').textContent =
          `Session: ${fmtTime(elapsed * 1000)} elapsed \u00B7 started ${new Date(sessionStart).toLocaleTimeString()}`;
      }, 1000);
      $('session-info').textContent = `Session: 0s elapsed \u00B7 started ${new Date(sessionStart).toLocaleTimeString()}`;

      loop();
    } catch (err) {
      console.error('[ErgoSentinel] Session start failed:', err);
      loading.style.display = 'block';
      loading.textContent = 'Failed to start the session: ' + err.message;
      btnStart.disabled = false;
    }
  }

  function stopSession() {
    state.running = false;
    if (state.rafId) cancelAnimationFrame(state.rafId);
    if (state.sessionTicker) { clearInterval(state.sessionTicker); state.sessionTicker = null; }
    if (state.stream) state.stream.getTracks().forEach(tr => tr.stop());
    state.stream = null;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    modal.classList.remove('show');        // NEW-01b: dismiss modal on stop
    loading.style.display = 'block';
    loading.textContent = 'Session ended. Click Start to begin again.';
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnCal.disabled = true;
    btnExp.disabled = (state.history.length === 0);
    pushAlert('good', `Session ended. Smoothed strain: ${Math.round(state.smoothedStrain)}. Samples: ${state.history.length}.`, null);
  }

  function exportReport() {
    const wallNow = Date.now();
    const durationSec = state.sessionStartWall ? Math.round((wallNow - state.sessionStartWall) / 1000) : 0;

    const report = {
      product: 'ErgoSentinel',
      version: '0.4.8-production',
      buildNote: 'v0.4.8: display fixes only (Detection% excludes hidden hips; frame-ancestors removed). Metrics unchanged.',
      poseModel: 'MoveNet SinglePose Lightning (TensorFlow.js)',
      demoMode: demoMode,
      generated: new Date(wallNow).toISOString(),
      sessionStartedISO: state.sessionStartWall ? new Date(state.sessionStartWall).toISOString() : null,  // NEW-03
      sessionDurationSec: durationSec,                                 // wall-clock (INCLUDES hidden-tab gaps)
      monitoredSec: Math.round(state.monitoredMs / 1000),              // v0.4.6: time the loop actually ran — USE THIS as the denominator for exposure ratios
      poorPostureSec: Math.round(state.poorPostureMs / 1000),
      microBreaksTaken: state.breaksTaken,
      finalSmoothedStrain: Math.round(state.smoothedStrain),          // Fix-2 (was finalStrainIndex)
      validSampleCount: state.history.filter(s => s.risk !== null).length,   // GAP-08: distinguishes "no data" from "no strain"
      meanValidFrameFraction: state.history.length
        ? +(state.history.reduce((acc, s) => acc + (s.validFrameFraction || 0), 0) / state.history.length).toFixed(3)
        : 0,                                                           // GAP-08: session-level data-quality signal
      gateRejectionReasons: state.reasonCounts,                        // v0.4.2: WHY frames were invalid (diagnostics)
      performance: {                                                   // v0.4.6: evidences the O3 frame-rate claim
        backendUsed: state.backendUsed,                                // 'webgl' | 'cpu' — proves the configuration
        frameTimeMs: frameTimeStats(state.frameTimesMs),               // {n, p50, p95, p99, max, meanFps}
      },
      calibrationResetsDuringSession: state.calibrationResetCount,    // BUG-10 audit flag
      strainTauSec: TH.strainTauSec,                                  // Fix-1 provenance
      measurementScope: 'coronal-only (lateral head tilt + shoulder asymmetry); '
                      + 'sagittal FHP/lumbar/protraction out of scope',
      thresholds: TH,                                                 // NEW-04: no dead field
      noiseFloor: NOISE,
      samples: state.history,
      alerts: state.alerts.map(a => ({
        timestamp: a.wallClock ? new Date(a.wallClock).toISOString() : null,
        level: a.level,
        msg: a.msg
      }))
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ergosentinel-session-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // ---------- Events ----------
  // BUG-02: defensive startup guard — nothing but Start is actionable on load.
  btnStop.disabled = true;
  btnCal.disabled = true;
  btnExp.disabled = true;

  btnStart.addEventListener('click', startSession);
  btnStop.addEventListener('click', stopSession);
  btnCal.addEventListener('click', () => {
    // BUG-10: the reset is now audited and warned about, never silent.
    state.smoothedStrain = 0;
    state.poorPostureMs = 0;
    state.calibrationResetCount += 1;
    pushAlert('warn', 'Score reset. Smoothed strain reflects data from this point only.', null);
  });
  btnExp.addEventListener('click', exportReport);
  btnBreak.addEventListener('click', () => {
    modal.classList.remove('show');
    state.breaksTaken += 1;
    state.sittingActiveMs = 0;              // Fix-7: reset the presence-gated clock
    state.awayMs = 0;
    pushAlert('good', 'Micro-break recorded. Sitting timer reset.', null);
  });

  // ---------- Page Visibility (GAP-04/05 root-cause fix) ----------
  // Browsers PAUSE requestAnimationFrame in hidden/background tabs (MDN; Chrome
  // since 2011). Without handling, the first frame after the tab becomes visible
  // again carries a huge dt. We re-baseline every elapsed-time reference on
  // resume so no backgrounded gap is absorbed into the strain signal or the
  // sitting/poor-posture timers. The dt clamp in loop() is the defence in depth.
  document.addEventListener('visibilitychange', () => {
    if (!state.running) return;
    if (document.hidden) {
      state.paused = true;                 // rAF self-suspends; nothing accrues while hidden
    } else if (state.paused) {
      const tt = now();
      state.lastT = tt;                    // small dt on the first resumed frame
      state.lastSampleMs = tt;             // next sample window starts fresh
      state.lastPoorTickMs = null;         // do not accrue across the gap
      state.paused = false;
    }
  });

  // Render empty state on load
  renderAlerts();

  // ---------- Debug hook (no-op in production unless explicitly enabled) ----------
  if (typeof window !== 'undefined' && window.__ERGO_DEBUG) {
    window.__ergo = {
      state, TH, NOISE, GEOM, KP,
      extractFeatures, scorePosture, assessCaptureGeometry,
      updateSmoothedStrain, tickPresence, tickPoorPosture, maybeSample, resetSessionState,
      frameTimeStats,
    };
  }
})();
