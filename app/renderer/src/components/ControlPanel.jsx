import React, { useState } from "react";

export default function ControlPanel({
  profiles,
  scenarios,
  running,
  onStart,
  mode,
  onModeChange,
  paused,
  onPause,
  onResume,
  speed,
  onSpeedChange,
  classId,
  managedCount,
  totalAdhd,
}) {
  const [profile, setProfile] = useState("");
  const [scenario, setScenario] = useState("");

  const handleStart = () => {
    if (mode === "multi" || mode === "v2") {
      onStart(profile, scenario);
    } else if (profile && scenario) {
      onStart(profile, scenario);
    }
  };

  const profileList = profiles.length > 0 ? profiles : [
    { name: "mild_inattentive", label: "Mild Inattentive (8y)" },
    { name: "moderate_combined", label: "Moderate Combined (9y)" },
    { name: "severe_hyperactive", label: "Severe Hyperactive (11y)" },
  ];

  const scenarioList = scenarios.length > 0 ? scenarios : [
    { name: "recess_to_math", label: "Recess -> Math" },
    { name: "art_time_ending", label: "Art Time Ending" },
    { name: "surprise_assembly", label: "Surprise Assembly" },
    { name: "reading_to_cleanup", label: "Reading -> Cleanup" },
  ];

  const canStart = (mode === "multi" || mode === "v2") ? true : (!!profile && !!scenario);

  return (
    <div style={styles.container}>
      <div style={styles.title}>Simulation Control</div>

      {/* V2 mode only — classic/multi removed */}

      {/* Classic-only: profile & scenario selectors */}
      {mode === "classic" && (
        <>
          <div style={styles.field}>
            <label style={styles.label}>Profile</label>
            <select
              style={styles.select}
              value={profile}
              onChange={(e) => setProfile(e.target.value)}
              disabled={running}
            >
              <option value="">Select profile...</option>
              {profileList.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.label || p.name}
                </option>
              ))}
            </select>
          </div>

          <div style={styles.field}>
            <label style={styles.label}>Scenario</label>
            <select
              style={styles.select}
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              disabled={running}
            >
              <option value="">Select scenario...</option>
              {scenarioList.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.label || s.name}
                </option>
              ))}
            </select>
          </div>
        </>
      )}

      {/* Multi mode info */}
      {mode === "multi" && (
        <div style={styles.multiInfo}>
          <span style={styles.multiInfoText}>20명 학생 · ADHD 9% · 자동 생성</span>
          {classId != null && running && (
            <span style={styles.classCounter}>Class #{classId}</span>
          )}
        </div>
      )}

      {/* V2 mode info */}
      {mode === "v2" && (
        <div style={styles.multiInfo}>
          <span style={styles.multiInfoText}>1년 시뮬레이션 · 950턴 · 자동 아키타입</span>
          {classId != null && running && (
            <span style={styles.classCounter}>Class #{classId}</span>
          )}
        </div>
      )}

      {/* Multi/V2 mode: managed/total progress */}
      {(mode === "multi" || mode === "v2") && running && totalAdhd != null && (
        <div style={styles.field}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <span style={styles.label}>ADHD 관리 진행</span>
            <span style={{ fontSize: 11, color: "#a855f7", fontFamily: "monospace" }}>
              {managedCount ?? 0}/{totalAdhd}
            </span>
          </div>
          <div style={styles.progressBg}>
            <div
              style={{
                ...styles.progressFill,
                width: totalAdhd > 0 ? `${((managedCount ?? 0) / totalAdhd) * 100}%` : "0%",
              }}
            />
          </div>
        </div>
      )}

      {/* Speed slider (multi/v2 mode) */}
      {(mode === "multi" || mode === "v2") && running && (
        <div style={styles.field}>
          <label style={styles.label}>
            Speed: {speed != null ? `${speed.toFixed(1)}s/turn` : "1.0s/turn"}
          </label>
          <input
            type="range"
            min="0.3"
            max="5.0"
            step="0.1"
            value={speed ?? 1.5}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            style={styles.slider}
          />
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={styles.sliderLabel}>빠름 0.3s</span>
            <span style={styles.sliderLabel}>느림 5.0s</span>
          </div>
        </div>
      )}

      {/* Start / Pause+Resume row */}
      <div style={{ display: "flex", gap: 6, marginTop: 4 }}>
        <button
          style={{
            ...styles.button,
            flex: 1,
            opacity: running || !canStart ? 0.5 : 1,
          }}
          onClick={handleStart}
          disabled={running || !canStart}
        >
          {running ? "Running..." : "Start Session"}
        </button>

        {(mode === "multi" || mode === "v2") && running && (
          <button
            style={styles.pauseBtn}
            onClick={paused ? onResume : onPause}
          >
            {paused ? "▶" : "⏸"}
          </button>
        )}
      </div>
    </div>
  );
}

const styles = {
  container: {
    background: "#1e293b",
    borderRadius: 8,
    padding: 12,
    border: "1px solid #334155",
  },
  title: {
    fontSize: 12,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 10,
  },
  modeRow: {
    display: "flex",
    gap: 4,
    marginBottom: 10,
  },
  modeBtn: {
    flex: 1,
    padding: "5px 0",
    background: "#0f172a",
    color: "#64748b",
    border: "1px solid #334155",
    borderRadius: 5,
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
  },
  modeBtnActive: {
    background: "#1e3a5f",
    color: "#60a5fa",
    borderColor: "#3b82f6",
  },
  field: {
    marginBottom: 8,
  },
  label: {
    display: "block",
    fontSize: 11,
    color: "#94a3b8",
    marginBottom: 4,
  },
  select: {
    width: "100%",
    padding: "6px 8px",
    background: "#0f172a",
    color: "#e2e8f0",
    border: "1px solid #334155",
    borderRadius: 4,
    fontSize: 13,
    outline: "none",
  },
  multiInfo: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    background: "#0f172a",
    borderRadius: 5,
    padding: "6px 8px",
    marginBottom: 8,
  },
  multiInfoText: {
    fontSize: 11,
    color: "#64748b",
  },
  classCounter: {
    fontSize: 11,
    color: "#60a5fa",
    fontWeight: 700,
    fontFamily: "monospace",
  },
  progressBg: {
    height: 6,
    background: "#0f172a",
    borderRadius: 3,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "#a855f7",
    borderRadius: 3,
    transition: "width 0.4s ease",
  },
  slider: {
    width: "100%",
    accentColor: "#3b82f6",
    cursor: "pointer",
  },
  sliderLabel: {
    fontSize: 9,
    color: "#475569",
  },
  button: {
    padding: "8px 0",
    background: "#3b82f6",
    color: "#fff",
    border: "none",
    borderRadius: 6,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
  },
  pauseBtn: {
    padding: "8px 12px",
    background: "#334155",
    color: "#e2e8f0",
    border: "none",
    borderRadius: 6,
    fontSize: 14,
    cursor: "pointer",
  },
};
