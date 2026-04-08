import React, { useEffect, useRef } from "react";
import Phaser from "phaser";

/*
  Unified school map with walking character sprites.
  Canvas: 800x750
    [0,   0] - [800, 620]: Reference image (classroom_bg.png scaled)
    [0, 620] - [800, 750]: Playground (drawn programmatically)

  The reference image contains classroom (top-left), office (top-right),
  hallway with lockers (bottom). The playground seamlessly extends below.
*/

// ── Exact color palette from the reference image ─────────────────────────
const PAL = {
  border: 0x3b3558,
  gold: 0x8b7a4a,
  white: 0xffffff,
  cream: 0xf0e0c0,
  floorLight: 0xc8b89a,
  floorDark: 0xbaa882,
  deskWood: 0xc49560,
  deskEdge: 0x9a7040,
  chairGreen: 0x2d6e5a,
  chairDark: 0x1e4d3d,
  lockerBlue: 0x8899bb,
  lockerDark: 0x7788aa,
  bookshelf: 0x8b6530,
  windowGlass: 0x88bbdd,
  windowFrame: 0xddccaa,
  curtainRed: 0xaa3333,
  doorBrown: 0x9a7040,
  grassA: 0x6a9a3a,
  grassB: 0x5a8a2a,
  treeLeafA: 0x4a8a3a,
  treeLeafB: 0x3a7a2a,
  trunk: 0x8b6530,
  sandbox: 0xd8c8a0,
  hoopPole: 0x8899bb,
  hoopRim: 0xd08040,
  skinTone: 0xf0c8a0,
  pants: 0x334455,
  eyes: 0x222222,
};

// ── Area bounding boxes on 800x750 ───────────────────────────────────────
const AREAS = {
  classroom:  { x: 0,   y: 0,   w: 530, h: 440, label: "교실",   color: 0x60a5fa },
  office:     { x: 530, y: 0,   w: 270, h: 440, label: "교무실", color: 0xa855f7 },
  hallway:    { x: 0,   y: 440, w: 800, h: 180, label: "복도",   color: 0x4ade80 },
  playground: { x: 0,   y: 620, w: 800, h: 130, label: "운동장", color: 0xf59e0b },
};

// ── Student/teacher positions per area ───────────────────────────────────
const AREA_POSITIONS = {
  classroom: {
    desks: [
      { x: 168, y: 210 }, { x: 248, y: 210 }, { x: 328, y: 190 },
      { x: 140, y: 290 }, { x: 220, y: 290 }, { x: 300, y: 290 }, { x: 380, y: 290 },
      { x: 140, y: 370 }, { x: 220, y: 370 }, { x: 300, y: 370 }, { x: 380, y: 370 },
      { x: 140, y: 440 }, { x: 220, y: 440 }, { x: 300, y: 440 }, { x: 380, y: 440 },
      { x: 168, y: 510 }, { x: 248, y: 510 }, { x: 328, y: 510 }, { x: 408, y: 510 },
      { x: 248, y: 150 },
    ],
    teacher: { x: 280, y: 130 },
  },
  office: {
    seats: [{ x: 600, y: 280 }, { x: 680, y: 320 }],
    teacher: { x: 700, y: 240 },
  },
  hallway: {
    path: [
      { x: 100, y: 540 }, { x: 200, y: 540 }, { x: 300, y: 540 },
      { x: 400, y: 540 }, { x: 500, y: 540 }, { x: 600, y: 540 },
      { x: 150, y: 570 }, { x: 250, y: 570 }, { x: 350, y: 570 },
      { x: 450, y: 570 }, { x: 550, y: 570 }, { x: 650, y: 570 },
      { x: 100, y: 600 }, { x: 200, y: 600 }, { x: 300, y: 600 },
      { x: 400, y: 600 }, { x: 500, y: 600 }, { x: 600, y: 600 },
      { x: 700, y: 540 }, { x: 700, y: 600 },
    ],
    teacher: { x: 400, y: 560 },
  },
  playground: {
    scattered: [
      { x: 80, y: 660 }, { x: 180, y: 680 }, { x: 300, y: 650 }, { x: 420, y: 690 },
      { x: 550, y: 660 }, { x: 650, y: 680 }, { x: 120, y: 720 }, { x: 250, y: 730 },
      { x: 370, y: 710 }, { x: 500, y: 730 }, { x: 600, y: 710 }, { x: 700, y: 720 },
      { x: 150, y: 670 }, { x: 350, y: 700 }, { x: 480, y: 670 }, { x: 620, y: 700 },
      { x: 200, y: 700 }, { x: 440, y: 720 }, { x: 560, y: 690 }, { x: 680, y: 660 },
    ],
    teacher: { x: 400, y: 680 },
  },
};

// ── Character profiles ───────────────────────────────────────────────────
const CHARACTER_PROFILES = [
  { id: "teacher", hair: 0x553320, shirt: 0xeeeeee, isTeacher: true },
  { id: "S01", hair: 0x553320, shirt: 0x4477aa },
  { id: "S02", hair: 0x886640, shirt: 0x44aa77 },
  { id: "S03", hair: 0x332244, shirt: 0xaa6644 },
  { id: "S04", hair: 0x221122, shirt: 0x5588bb },
  { id: "S05", hair: 0x664422, shirt: 0x66aa55 },
  { id: "S06", hair: 0x443322, shirt: 0xbb5555 },
  { id: "S07", hair: 0x554433, shirt: 0x7766aa },
  { id: "S08", hair: 0x332211, shirt: 0x44bbaa },
  { id: "S09", hair: 0x665544, shirt: 0xaa7744 },
  { id: "S10", hair: 0x222233, shirt: 0x5599aa },
  { id: "S11", hair: 0x776655, shirt: 0x448866 },
  { id: "S12", hair: 0x443355, shirt: 0xbb7766 },
  { id: "S13", hair: 0x554422, shirt: 0x6688bb },
  { id: "S14", hair: 0x332233, shirt: 0x77aa55 },
  { id: "S15", hair: 0x665533, shirt: 0xaa5577 },
  { id: "S16", hair: 0x443311, shirt: 0x55aaaa },
  { id: "S17", hair: 0x221133, shirt: 0x88bb55 },
  { id: "S18", hair: 0x554411, shirt: 0xbb8844 },
  { id: "S19", hair: 0x333344, shirt: 0x5577bb },
  { id: "S20", hair: 0x664433, shirt: 0x44aa88 },
];

// ── Derive active area from scenario + action ────────────────────────────
function deriveArea(scenario, action, turn, totalTurns) {
  if (action === "private_correction") return "office";
  if (!scenario) return "classroom";
  if (scenario === "recess_to_math") {
    const pivot = totalTurns ? totalTurns / 2 : 15;
    if (turn <= pivot) return "playground";
    if (turn <= pivot + 2) return "hallway";
    return "classroom";
  }
  return "classroom";
}

// ── Student risk color ───────────────────────────────────────────────────
function studentColor(student) {
  if (student.is_managed) return 0xa855f7;
  if (student.is_identified) return 0x60a5fa;
  const s = student.state || {};
  const esc = s.escalation ?? s.escalation_risk ?? 0;
  const dis = s.distress ?? s.distress_level ?? 0;
  if (esc > 0.6 || dis > 0.7) return 0xef4444;
  if (esc > 0.35 || dis > 0.45) return 0xf59e0b;
  return 0x4ade80;
}

// ── Draw a single pixel-art character ────────────────────────────────────
function drawCharacter(g, x, y, hairColor, shirtColor, facing, frame, isTeacher) {
  const scale = isTeacher ? 1.25 : 1.0;
  const sx = (v) => v * scale;

  // Shadow
  g.fillStyle(0x000000, 0.15);
  g.fillEllipse(x, y + sx(11), sx(12), sx(4));

  // Body (shirt)
  g.fillStyle(shirtColor);
  g.fillRect(x - sx(4), y + sx(2), sx(8), sx(7));

  // Legs (walk frame)
  g.fillStyle(PAL.pants);
  if (frame === 0) {
    g.fillRect(x - sx(3), y + sx(9), sx(3), sx(3));
    g.fillRect(x + sx(1), y + sx(9), sx(3), sx(3));
  } else {
    g.fillRect(x - sx(2), y + sx(9), sx(3), sx(3));
    g.fillRect(x, y + sx(9), sx(3), sx(3));
  }

  // Head (skin)
  g.fillStyle(PAL.skinTone);
  g.fillRect(x - sx(4), y - sx(6), sx(8), sx(8));

  // Hair
  g.fillStyle(hairColor);
  if (facing === "up") {
    g.fillRect(x - sx(4), y - sx(6), sx(8), sx(4));
    g.fillRect(x - sx(4), y - sx(6), sx(2), sx(6));
    g.fillRect(x + sx(2), y - sx(6), sx(2), sx(6));
  } else if (facing === "left") {
    g.fillRect(x - sx(4), y - sx(6), sx(8), sx(4));
    g.fillRect(x - sx(4), y - sx(6), sx(3), sx(7));
  } else if (facing === "right") {
    g.fillRect(x - sx(4), y - sx(6), sx(8), sx(4));
    g.fillRect(x + sx(1), y - sx(6), sx(3), sx(7));
  } else {
    // down (default)
    g.fillRect(x - sx(4), y - sx(6), sx(8), sx(4));
    g.fillRect(x - sx(4), y - sx(6), sx(2), sx(6));
    g.fillRect(x + sx(2), y - sx(6), sx(2), sx(6));
  }

  // Eyes (visible when facing down, left, right)
  if (facing !== "up") {
    g.fillStyle(PAL.eyes);
    const eyeOffX = facing === "left" ? -1 : facing === "right" ? 1 : 0;
    g.fillRect(x - sx(2) + eyeOffX, y - sx(2), sx(1.5), sx(1.5));
    g.fillRect(x + sx(1) + eyeOffX, y - sx(2), sx(1.5), sx(1.5));
  }

  // Teacher marker: small white collar
  if (isTeacher) {
    g.fillStyle(0xffffff, 0.7);
    g.fillRect(x - sx(2), y + sx(1), sx(4), sx(1.5));
  }
}

// ── Facing direction from movement vector ────────────────────────────────
function getFacing(dx, dy) {
  if (Math.abs(dx) > Math.abs(dy)) {
    return dx > 0 ? "right" : "left";
  }
  return dy > 0 ? "down" : "up";
}

// ── Phaser Scene ─────────────────────────────────────────────────────────
class UnifiedSchoolScene extends Phaser.Scene {
  constructor() {
    super("UnifiedSchoolScene");
    this.mode = "classic";
    this.students = [];
    this.targetStudentId = null;
    this.activeArea = "classroom";

    // Graphics layers
    this.bgImage = null;
    this.playgroundGraphics = null;
    this.overlayGraphics = null;
    this.highlightGraphics = null;
    this.characterGraphics = null;

    // Character state: map id -> { x, y, targetX, targetY, hair, shirt, facing, frame, frameCounter, isTeacher }
    this.characters = new Map();
    this._frameCount = 0;

    // Interaction lines (v2): [{x1,y1,x2,y2,color,text,alpha,createdAt}]
    this._interactionLines = [];
    this.interactionGraphics = null;

    // UI
    this.statusText = null;
    this.turnText = null;
    this.actionText = null;
    this.actionBg = null;
    this.bubbleGroup = null;
    this.areaLabelTexts = {};
  }

  preload() {
    this.load.image("classroom", "/assets/classroom_bg.png");
    this.load.image("playground", "/assets/playground_bg.png");
  }

  create() {
    const W = this.scale.width;   // 800
    const H = this.scale.height;  // 750

    // ── Layer 0: Background image (top 620px) ──
    this.bgImage = this.add.image(0, 0, "classroom").setOrigin(0, 0);
    this.bgImage.setDisplaySize(800, 620);

    // ── Layer 1: Programmatic playground (y=620 to y=750) ──
    this.playgroundGraphics = this.add.graphics();
    this._drawPlayground();

    // ── Layer 2: Dim overlays per area ──
    this.overlayGraphics = this.add.graphics();

    // ── Layer 3: Active area highlight glow ──
    this.highlightGraphics = this.add.graphics();

    // ── Layer 4: Character sprites (redrawn each frame) ──
    this.characterGraphics = this.add.graphics();

    // ── Layer 4.5: Interaction lines (v2) ──
    this.interactionGraphics = this.add.graphics();

    // ── Layer 5: Bubble group ──
    this.bubbleGroup = this.add.group();

    // ── Layer 6: Area labels ──
    this._createAreaLabels();

    // ── HUD: Status bar ──
    this.statusBg = this.add.graphics();
    this.statusBg.fillStyle(0x1e293b, 0.88);
    this.statusBg.fillRoundedRect(8, H - 42, 220, 30, 5);

    this.statusText = this.add.text(16, H - 36, "Waiting for session...", {
      fontSize: "10px",
      fontFamily: "'Pretendard', monospace",
      color: "#94a3b8",
    });

    // ── HUD: Turn counter ──
    this.turnBg = this.add.graphics();
    this.turnBg.fillStyle(0x1e293b, 0.88);
    this.turnBg.fillRoundedRect(W - 90, 8, 82, 22, 4);

    this.turnText = this.add.text(W - 82, 13, "", {
      fontSize: "10px",
      fontFamily: "monospace",
      color: "#e2e8f0",
    });

    // ── HUD: Action label ──
    this.actionBg = this.add.graphics();
    this.actionText = this.add.text(W / 2, 10, "", {
      fontSize: "11px",
      fontFamily: "'Pretendard', monospace",
      color: "#fbbf24",
      fontStyle: "bold",
    }).setOrigin(0.5, 0);

    // ── Transition label ──
    this.transitionLabel = this.add.text(W / 2, H / 2, "", {
      fontSize: "18px",
      fontFamily: "'Pretendard', monospace",
      color: "#ffffff",
      fontStyle: "bold",
      stroke: "#000000",
      strokeThickness: 3,
    }).setOrigin(0.5, 0.5).setAlpha(0);

    // Initialize characters at classroom positions
    this._initCharacters();

    // Draw initial overlays
    this._drawOverlaysAndHighlight("classroom");
  }

  // ── Initialize character sprites at default positions ──────────────────
  _initCharacters() {
    const desks = AREA_POSITIONS.classroom.desks;
    const teacherPos = AREA_POSITIONS.classroom.teacher;

    // Teacher
    const tp = CHARACTER_PROFILES[0];
    this.characters.set(tp.id, {
      x: teacherPos.x, y: teacherPos.y,
      targetX: teacherPos.x, targetY: teacherPos.y,
      hair: tp.hair, shirt: tp.shirt,
      facing: "down", frame: 0, frameCounter: 0,
      isTeacher: true,
    });

    // Students
    for (let i = 1; i < CHARACTER_PROFILES.length; i++) {
      const p = CHARACTER_PROFILES[i];
      const pos = desks[(i - 1) % desks.length];
      this.characters.set(p.id, {
        x: pos.x, y: pos.y,
        targetX: pos.x, targetY: pos.y,
        hair: p.hair, shirt: p.shirt,
        facing: "down", frame: 0, frameCounter: 0,
        isTeacher: false,
      });
    }
  }

  // ── Per-frame update: move characters toward targets + redraw ──────────
  update() {
    this._frameCount++;
    const SPEED = 2;
    let anyMoved = false;

    this.characters.forEach((ch) => {
      const dx = ch.targetX - ch.x;
      const dy = ch.targetY - ch.y;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist > SPEED) {
        // Move toward target
        ch.x += (dx / dist) * SPEED;
        ch.y += (dy / dist) * SPEED;
        ch.facing = getFacing(dx, dy);
        anyMoved = true;

        // Walk animation: toggle frame every 8 game frames
        ch.frameCounter++;
        if (ch.frameCounter >= 8) {
          ch.frame = ch.frame === 0 ? 1 : 0;
          ch.frameCounter = 0;
        }
      } else {
        // Snap to target, idle
        ch.x = ch.targetX;
        ch.y = ch.targetY;
        ch.frameCounter = 0;
      }
    });

    // Redraw characters every frame (they layer on top of everything)
    this._drawAllCharacters();

    // Draw and fade interaction lines
    this._drawInteractionLines();
  }

  // ── Draw all characters ────────────────────────────────────────────────
  _drawAllCharacters() {
    if (!this.characterGraphics) return;
    this.characterGraphics.clear();
    const g = this.characterGraphics;

    // Sort by y for depth ordering
    const sorted = [...this.characters.values()].sort((a, b) => a.y - b.y);

    sorted.forEach((ch) => {
      drawCharacter(
        g, ch.x, ch.y,
        ch.hair, ch.shirt,
        ch.facing, ch.frame,
        ch.isTeacher
      );

      // Status indicator dot above head
      if (ch._simData) {
        const statusCol = studentColor(ch._simData);
        const size = ch.isTeacher ? 1.25 : 1.0;
        g.fillStyle(statusCol, 0.9);
        g.fillCircle(ch.x, ch.y - 8 * size - 4, 3);
      }

      // ADHD identified ring
      if (ch._simData?.is_identified) {
        const size = ch.isTeacher ? 14 : 11;
        g.lineStyle(1.5, 0x60a5fa, 0.8);
        g.strokeCircle(ch.x, ch.y, size + 2);
      }

      // Managed checkmark dot
      if (ch._simData?.is_managed) {
        const size = ch.isTeacher ? 1.25 : 1.0;
        g.fillStyle(0xa855f7, 0.9);
        g.fillCircle(ch.x, ch.y - 8 * size - 4, 3);
      }
    });
  }

  // ── Draw playground below the reference image ──────────────────────────
  _drawPlayground() {
    const PY = 620;
    const PH = 130;
    const W = 800;

    // Use the provided pixel art soccer field image as playground background
    if (this.textures.exists("playground")) {
      if (this.playgroundImage) this.playgroundImage.destroy();
      this.playgroundImage = this.add.image(0, PY, "playground").setOrigin(0, 0);
      this.playgroundImage.setDisplaySize(W, PH);
      // Move below overlays but above background
      this.playgroundImage.setDepth(0.5);
    } else {
      // Fallback: simple green field
      const g = this.playgroundGraphics;
      g.fillStyle(0x4a8a3a);
      g.fillRect(0, PY, W, PH);
    }
  }

  // ── Area labels ────────────────────────────────────────────────────────
  _createAreaLabels() {
    const labelDefs = [
      { key: "classroom",  x: 16,  y: 16,  text: "교실" },
      { key: "office",     x: 540, y: 16,  text: "교무실" },
      { key: "hallway",    x: 16,  y: 448, text: "복도" },
      { key: "playground", x: 16,  y: 628, text: "운동장" },
    ];
    labelDefs.forEach(({ key, x, y, text }) => {
      const bg = this.add.graphics();
      bg.fillStyle(0x1e293b, 0.82);
      bg.fillRoundedRect(x - 4, y - 3, text.length * 11 + 14, 22, 4);
      bg.lineStyle(1, 0x334155, 0.7);
      bg.strokeRoundedRect(x - 4, y - 3, text.length * 11 + 14, 22, 4);
      const t = this.add.text(x + 3, y + 8, text, {
        fontSize: "11px",
        fontFamily: "'Pretendard', monospace",
        color: "#e2e8f0",
        fontStyle: "bold",
      }).setOrigin(0, 0.5);
      this.areaLabelTexts[key] = { bg, t };
    });
  }

  // ── Overlay + highlight ────────────────────────────────────────────────
  _drawOverlaysAndHighlight(activeArea) {
    if (!this.overlayGraphics) return;
    this.overlayGraphics.clear();
    this.highlightGraphics.clear();

    const DIM = 0.35;

    Object.entries(AREAS).forEach(([key, area]) => {
      if (key === activeArea) return;
      this.overlayGraphics.fillStyle(PAL.border, DIM);
      this.overlayGraphics.fillRect(area.x, area.y, area.w, area.h);
    });

    // Glow border on active area
    const active = AREAS[activeArea];
    if (active) {
      const hg = this.highlightGraphics;
      const c = active.color;
      hg.lineStyle(3, c, 0.85);
      hg.strokeRect(active.x + 2, active.y + 2, active.w - 4, active.h - 4);
      hg.lineStyle(1, c, 0.35);
      hg.strokeRect(active.x + 6, active.y + 6, active.w - 12, active.h - 12);
      // Corner accents
      const cLen = 20;
      hg.lineStyle(4, c, 0.9);
      hg.strokeLine(active.x + 2, active.y + 2, active.x + 2 + cLen, active.y + 2);
      hg.strokeLine(active.x + 2, active.y + 2, active.x + 2, active.y + 2 + cLen);
      hg.strokeLine(active.x + active.w - 2, active.y + 2, active.x + active.w - 2 - cLen, active.y + 2);
      hg.strokeLine(active.x + active.w - 2, active.y + 2, active.x + active.w - 2, active.y + 2 + cLen);
      hg.strokeLine(active.x + 2, active.y + active.h - 2, active.x + 2 + cLen, active.y + active.h - 2);
      hg.strokeLine(active.x + 2, active.y + active.h - 2, active.x + 2, active.y + active.h - 2 - cLen);
      hg.strokeLine(active.x + active.w - 2, active.y + active.h - 2, active.x + active.w - 2 - cLen, active.y + active.h - 2);
      hg.strokeLine(active.x + active.w - 2, active.y + active.h - 2, active.x + active.w - 2, active.y + active.h - 2 - cLen);
    }

    // Label brightness
    Object.entries(this.areaLabelTexts).forEach(([key, { t }]) => {
      t.setAlpha(key === activeArea ? 1.0 : 0.55);
    });
  }

  // ── Set active area with transition ────────────────────────────────────
  setActiveArea(area) {
    const prev = this.activeArea;
    this.activeArea = area;
    if (prev !== area) {
      const fromName = AREAS[prev]?.label || prev;
      const toName = AREAS[area]?.label || area;
      if (this.transitionLabel) {
        this.transitionLabel.setText(`${fromName} → ${toName}`).setAlpha(1);
        this.tweens.add({
          targets: this.transitionLabel,
          alpha: 0,
          duration: 1600,
          delay: 700,
          ease: "Power2",
        });
      }
    }
    this._drawOverlaysAndHighlight(area);
  }

  // ── Assign character targets based on area ─────────────────────────────
  _assignTargets(activeArea, targetId) {
    const areaPos = AREA_POSITIONS[activeArea];
    if (!areaPos) return;

    // Teacher target
    const teacherCh = this.characters.get("teacher");
    if (teacherCh) {
      const tPos = areaPos.teacher || AREA_POSITIONS.classroom.teacher;
      teacherCh.targetX = tPos.x;
      teacherCh.targetY = tPos.y;
    }

    // If teacher is observing a student, move toward them
    if (targetId && teacherCh) {
      const targetCh = this.characters.get(targetId);
      if (targetCh) {
        teacherCh.targetX = targetCh.targetX + 20;
        teacherCh.targetY = targetCh.targetY - 10;
      }
    }

    // Student targets
    const studentProfiles = CHARACTER_PROFILES.slice(1);
    studentProfiles.forEach((p, i) => {
      const ch = this.characters.get(p.id);
      if (!ch) return;

      // If this student goes to office for private_correction
      if (activeArea === "office" && p.id === targetId) {
        const officeSeats = AREA_POSITIONS.office.seats;
        ch.targetX = officeSeats[0].x;
        ch.targetY = officeSeats[0].y;
        return;
      }

      let positions;
      switch (activeArea) {
        case "classroom":
          positions = areaPos.desks;
          break;
        case "hallway":
          positions = areaPos.path;
          break;
        case "playground":
          positions = areaPos.scattered;
          break;
        default:
          positions = AREA_POSITIONS.classroom.desks;
      }

      const pos = positions[i % positions.length];
      ch.targetX = pos.x;
      ch.targetY = pos.y;
    });
  }

  // ── Map simulation student data to character profiles ──────────────────
  _mapStudentsToCharacters(simStudents) {
    // Assign simulation student ids to character profiles
    simStudents.forEach((student, idx) => {
      const profileIdx = idx + 1; // skip teacher
      if (profileIdx >= CHARACTER_PROFILES.length) return;
      const profileId = CHARACTER_PROFILES[profileIdx].id;
      const ch = this.characters.get(profileId);
      if (ch) {
        ch._simId = student.id;
        ch._simData = student;
      }
    });
  }

  // ── Interaction lines (v2) ──────────────────────────────────────────────
  showInteractionLines(interactions) {
    if (!interactions || !Array.isArray(interactions) || interactions.length === 0) return;
    const now = Date.now();

    interactions.forEach(({ actor, target, event_type }) => {
      // Find character profile ids that map to these simulation student ids
      let actorChId = null;
      let targetChId = null;
      this.characters.forEach((ch, id) => {
        if (ch._simId === actor) actorChId = id;
        if (ch._simId === target) targetChId = id;
      });
      if (!actorChId || !targetChId) return;

      const chA = this.characters.get(actorChId);
      const chB = this.characters.get(targetChId);
      if (!chA || !chB) return;

      // Color by interaction type
      let color = 0x94a3b8; // default gray
      if (event_type === "peer_conflict" || event_type === "peer_bullying") color = 0xef4444;
      else if (event_type === "peer_chat" || event_type === "peer_help" || event_type === "peer_friendship") color = 0x60a5fa;
      else if (event_type === "peer_contagion") color = 0xfbbf24;

      this._interactionLines.push({
        x1: chA.x, y1: chA.y,
        x2: chB.x, y2: chB.y,
        color,
        createdAt: now,
      });
    });
  }

  _drawInteractionLines() {
    if (!this.interactionGraphics) return;
    this.interactionGraphics.clear();
    const now = Date.now();
    const DURATION = 2000; // 2 seconds fade
    // Remove expired lines
    this._interactionLines = this._interactionLines.filter((l) => now - l.createdAt < DURATION);
    this._interactionLines.forEach((line) => {
      const elapsed = now - line.createdAt;
      const alpha = Math.max(0, 1 - elapsed / DURATION);
      this.interactionGraphics.lineStyle(2, line.color, alpha * 0.7);
      this.interactionGraphics.strokeLineShape(
        new Phaser.Geom.Line(line.x1, line.y1, line.x2, line.y2)
      );
      // Draw label at midpoint
      const mx = (line.x1 + line.x2) / 2;
      const my = (line.y1 + line.y2) / 2;
      this.interactionGraphics.fillStyle(line.color, alpha);
      this.interactionGraphics.fillCircle(mx, my, 3);
    });
  }

  // ── Bubble ─────────────────────────────────────────────────────────────
  showBubble(x, y, text, isTeacher) {
    this.bubbleGroup.clear(true, true);
    const maxWidth = 160;
    const padding = 6;
    const shortText = text.length > 60 ? text.substring(0, 57) + "..." : text;
    const bg = this.add.graphics();
    const textObj = this.add.text(x + padding, y + padding, shortText, {
      fontSize: "9px",
      fontFamily: "'Pretendard', sans-serif",
      color: isTeacher ? "#1e293b" : "#44403c",
      wordWrap: { width: maxWidth - padding * 2 },
      lineSpacing: 2,
    });
    const bounds = textObj.getBounds();
    const bw = Math.min(maxWidth, bounds.width + padding * 2 + 4);
    const bh = bounds.height + padding * 2 + 2;
    const bx = Math.max(4, Math.min(x, this.scale.width - bw - 4));
    const by = Math.max(4, Math.min(y, this.scale.height - bh - 8));
    textObj.setPosition(bx + padding + 2, by + padding);
    bg.fillStyle(isTeacher ? 0xffffff : 0xfef3c7, 0.95);
    bg.fillRoundedRect(bx, by, bw, bh, 4);
    bg.lineStyle(1, isTeacher ? 0x94a3b8 : 0xd97706, 0.5);
    bg.strokeRoundedRect(bx, by, bw, bh, 4);
    const tailX = Math.min(x + 12, bx + bw - 12);
    bg.fillStyle(isTeacher ? 0xffffff : 0xfef3c7, 0.95);
    bg.fillTriangle(tailX, by + bh, tailX + 6, by + bh, tailX + 3, by + bh + 6);
    this.bubbleGroup.add(bg);
    this.bubbleGroup.add(textObj);
    this.time.delayedCall(5000, () => { bg?.destroy(); textObj?.destroy(); });
  }

  // ── Classic mode update ────────────────────────────────────────────────
  updateSimState(state, events, scenario) {
    const lastEvent = events[events.length - 1];
    const turn = lastEvent?.time ?? 0;
    const action = lastEvent?.action || "";
    const area = deriveArea(scenario, action, turn, null);

    if (area !== this.activeArea) {
      this.setActiveArea(area);
    }

    // Move all characters to the active area positions
    this._assignTargets(area, null);

    if (state) {
      const risk = state.escalation_risk || 0;
      const compliance = state.compliance || 0;
      const distress = state.distress_level || 0;
      let statusLabel = "Calm", statusColor = "#60a5fa";
      if (risk > 0.7)           { statusLabel = "ESCALATING"; statusColor = "#ef4444"; }
      else if (distress > 0.6)  { statusLabel = "Distressed"; statusColor = "#f59e0b"; }
      else if (compliance > 0.7){ statusLabel = "Compliant"; statusColor = "#4ade80"; }
      if (this.statusText) {
        this.statusText.setText(
          `${statusLabel}  D:${(distress * 100).toFixed(0)}  C:${(compliance * 100).toFixed(0)}  A:${((state.attention || 0) * 100).toFixed(0)}  E:${(risk * 100).toFixed(0)}`
        );
        this.statusText.setColor(statusColor);
      }
    }

    if (lastEvent) {
      if (this.turnText && lastEvent.time !== undefined) {
        this.turnText.setText(`Turn ${lastEvent.time}`);
      }
      if (this.actionText && lastEvent.action) {
        const label = lastEvent.action.replace(/_/g, " ");
        this.actionText.setText(label);
        if (this.actionBg) {
          this.actionBg.clear();
          this.actionBg.fillStyle(0x1e293b, 0.8);
          const aw = label.length * 8 + 20;
          this.actionBg.fillRoundedRect(this.scale.width / 2 - aw / 2, 6, aw, 22, 4);
        }
      }
      if (lastEvent.utterance && lastEvent.speaker) {
        const teacherCh = this.characters.get("teacher");
        const bx = teacherCh ? teacherCh.x + 20 : 180;
        const by = teacherCh ? teacherCh.y - 30 : 60;
        this.showBubble(bx, by, `${lastEvent.speaker}: ${lastEvent.utterance}`, true);
      }
      if (lastEvent.student_narrative) {
        this.showBubble(200, 160, lastEvent.student_narrative, false);
      }
    }
  }

  // ── Multi + V2 mode update ──────────────────────────────────────────────
  updateMultiState(students, teacherAction, turn, classId, scenario, v2Location) {
    const actionType = teacherAction?.action_type || "";
    const targetId = teacherAction?.student_id || null;
    // V2 provides location directly; multi mode derives it from scenario
    const area = v2Location || deriveArea(scenario, actionType, turn, null);

    if (area !== this.activeArea) {
      this.setActiveArea(area);
    }

    // Map simulation students to character profiles
    this._mapStudentsToCharacters(students);

    // Find the character profile id that maps to the target simulation student
    let targetCharId = null;
    if (targetId) {
      this.characters.forEach((ch, id) => {
        if (ch._simId === targetId) targetCharId = id;
      });
    }

    // If private_correction, move target to office, teacher follows
    if (actionType === "private_correction" && targetCharId) {
      this._assignTargets("classroom", null); // everyone stays in classroom
      const targetCh = this.characters.get(targetCharId);
      const teacherCh = this.characters.get("teacher");
      if (targetCh) {
        targetCh.targetX = AREA_POSITIONS.office.seats[0].x;
        targetCh.targetY = AREA_POSITIONS.office.seats[0].y;
      }
      if (teacherCh) {
        teacherCh.targetX = AREA_POSITIONS.office.teacher.x;
        teacherCh.targetY = AREA_POSITIONS.office.teacher.y;
      }
    } else {
      this._assignTargets(area, targetCharId);
    }

    // If observe, move teacher toward targeted student
    if (actionType === "observe" && targetCharId) {
      const targetCh = this.characters.get(targetCharId);
      const teacherCh = this.characters.get("teacher");
      if (targetCh && teacherCh) {
        teacherCh.targetX = targetCh.targetX + 20;
        teacherCh.targetY = targetCh.targetY - 10;
      }
    }

    // Update shirt color based on student risk state
    students.forEach((student, idx) => {
      const profileIdx = idx + 1;
      if (profileIdx >= CHARACTER_PROFILES.length) return;
      const profileId = CHARACTER_PROFILES[profileIdx].id;
      const ch = this.characters.get(profileId);
      if (!ch) return;
      const color = studentColor(student);
      // Tint shirt slightly based on risk (blend with original)
      if (color === 0xef4444) {
        ch.shirt = 0xcc5555; // red-ish tint
      } else if (color === 0xf59e0b) {
        ch.shirt = 0xccaa44; // yellow-ish tint
      } else if (color === 0xa855f7) {
        ch.shirt = 0x9966cc; // purple managed
      } else {
        ch.shirt = CHARACTER_PROFILES[profileIdx].shirt; // original
      }
    });

    if (this.turnText) {
      this.turnText.setText(`T${turn} C${classId}`);
    }

    const targetStudent = students.find((s) => s.id === targetId);
    const label = actionType
      ? `${actionType.replace(/_/g, " ")}${targetStudent ? ` → ${targetStudent.name}` : ""}`
      : "";
    if (this.actionText) {
      this.actionText.setText(label);
      if (this.actionBg && label) {
        this.actionBg.clear();
        this.actionBg.fillStyle(0x1e293b, 0.8);
        const aw = label.length * 7 + 20;
        this.actionBg.fillRoundedRect(this.scale.width / 2 - aw / 2, 6, aw, 22, 4);
      }
    }

    const managedCount = students.filter((s) => s.is_managed).length;
    const identifiedCount = students.filter((s) => s.is_identified).length;
    if (this.statusText) {
      this.statusText.setText(`식별: ${identifiedCount}  관리: ${managedCount}  학생: ${students.length}`);
      this.statusText.setColor("#94a3b8");
    }
  }

  // ── V2 mode update with interactions ──────────────────────────────────
  updateV2State(turnData) {
    const { students, teacher_action, turn, class_id, location, interactions } = turnData;
    if (!students) return;
    this.updateMultiState(students, teacher_action, turn, class_id, null, location || "classroom");
    if (interactions && Array.isArray(interactions) && interactions.length > 0) {
      this.showInteractionLines(interactions);
    }
  }
}

// ── React component ──────────────────────────────────────────────────────
export default function ClassroomView({ state, events, mode, multiTurnData, scenario, v2Info }) {
  const containerRef = useRef(null);
  const gameRef = useRef(null);
  const sceneRef = useRef(null);

  useEffect(() => {
    if (gameRef.current || !containerRef.current) return;

    const config = {
      type: Phaser.AUTO,
      parent: containerRef.current,
      width: 800,
      height: 750,
      backgroundColor: "#3b3558",
      pixelArt: true,
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
      },
      scene: UnifiedSchoolScene,
    };

    gameRef.current = new Phaser.Game(config);

    const checkScene = setInterval(() => {
      const scene = gameRef.current?.scene?.getScene("UnifiedSchoolScene");
      if (scene && scene.statusText) {
        sceneRef.current = scene;
        clearInterval(checkScene);
      }
    }, 100);

    return () => {
      clearInterval(checkScene);
      gameRef.current?.destroy(true);
      gameRef.current = null;
    };
  }, []);

  // Classic mode updates
  useEffect(() => {
    if (sceneRef.current && mode === "classic") {
      sceneRef.current.updateSimState(state, events ?? [], scenario);
    }
  }, [state, events, mode, scenario]);

  // Multi + V2 mode updates
  useEffect(() => {
    if (sceneRef.current && (mode === "multi" || mode === "v2") && multiTurnData) {
      if (mode === "v2") {
        sceneRef.current.updateV2State(multiTurnData);
      } else {
        const { students, teacher_action, turn, class_id } = multiTurnData;
        if (students) {
          sceneRef.current.updateMultiState(students, teacher_action, turn, class_id, scenario);
        }
      }
    }
  }, [multiTurnData, mode, scenario]);

  return (
    <div style={styles.container}>
      <div ref={containerRef} style={styles.canvas} />
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    position: "relative",
    borderRadius: 8,
    overflow: "hidden",
    border: "1px solid #334155",
    background: "#3b3558",
  },
  canvas: {
    width: "100%",
    height: "100%",
  },
};
