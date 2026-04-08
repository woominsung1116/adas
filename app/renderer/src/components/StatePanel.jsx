import React from "react";

const DIMS = [
  { key: "distress_level", multiKey: "distress", label: "Distress", color: "#ef4444" },
  { key: "compliance",     multiKey: "compliance", label: "Compliance", color: "#4ade80" },
  { key: "attention",      multiKey: "attention",  label: "Attention",  color: "#60a5fa" },
  { key: "escalation_risk", multiKey: "escalation", label: "Escalation", color: "#f59e0b" },
];

const ACTION_COLORS = {
  observe: "#60a5fa",
  identify_adhd: "#fbbf24",
  individual_intervention: "#4ade80",
  class_instruction: "#94a3b8",
  correct: "#ef4444",
};

export default function StatePanel({ state, mode, multiState }) {
  if (mode === "multi" || mode === "v2") {
    return <MultiStatePanel multiState={multiState} mode={mode} />;
  }
  return <ClassicStatePanel state={state} />;
}

function ClassicStatePanel({ state }) {
  return (
    <div style={styles.container}>
      <div style={styles.title}>Child State</div>
      {DIMS.map((dim) => {
        const value = state?.[dim.key] ?? 0;
        return (
          <div key={dim.key} style={styles.row}>
            <span style={styles.label}>{dim.label}</span>
            <div style={styles.barBg}>
              <div style={{ ...styles.barFill, width: `${value * 100}%`, background: dim.color }} />
            </div>
            <span style={styles.value}>{(value * 100).toFixed(0)}%</span>
          </div>
        );
      })}
    </div>
  );
}

function MultiStatePanel({ multiState, mode }) {
  const { focusedStudent, teacherAction, identifiedCount, managedCount, totalAdhd, v2Info } = multiState || {};

  const focusedState = focusedStudent?.state;
  const actionColor = ACTION_COLORS[teacherAction?.action_type] || "#94a3b8";

  return (
    <div style={styles.container}>
      <div style={styles.title}>
        {focusedStudent ? `집중 학생: ${focusedStudent.name}` : "Child State"}
      </div>

      {/* Summary row */}
      <div style={styles.summaryRow}>
        <span style={styles.summaryItem}>
          식별됨: <b style={{ color: "#60a5fa" }}>{identifiedCount ?? 0}</b>/{totalAdhd ?? "?"} ADHD
        </span>
        <span style={styles.summaryItem}>
          관리됨: <b style={{ color: "#a855f7" }}>{managedCount ?? 0}</b>/{totalAdhd ?? "?"}
        </span>
      </div>

      {/* V2 context info */}
      {mode === "v2" && v2Info && (
        <div style={styles.v2InfoRow}>
          <span style={styles.summaryItem}>
            Day {v2Info.day} · {v2Info.period}교시
          </span>
          <span style={styles.summaryItem}>
            {v2Info.subject || "—"} · {v2Info.location}
          </span>
        </div>
      )}

      {/* Focused student state bars */}
      {focusedState
        ? DIMS.map((dim) => {
            const value = focusedState[dim.multiKey] ?? focusedState[dim.key] ?? 0;
            return (
              <div key={dim.key} style={styles.row}>
                <span style={styles.label}>{dim.label}</span>
                <div style={styles.barBg}>
                  <div style={{ ...styles.barFill, width: `${value * 100}%`, background: dim.color }} />
                </div>
                <span style={styles.value}>{(value * 100).toFixed(0)}%</span>
              </div>
            );
          })
        : <div style={styles.noFocus}>선생님 행동 대기 중...</div>
      }

      {/* Teacher action */}
      {teacherAction && (
        <div style={styles.actionBox}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
            <span style={{ ...styles.actionTag, background: actionColor + "33", color: actionColor }}>
              {teacherAction.action_type?.replace(/_/g, " ")}
            </span>
            {teacherAction.student_id && (
              <span style={styles.actionStudent}>{teacherAction.student_id}</span>
            )}
          </div>
          {teacherAction.reasoning && (
            <div style={styles.reasoning}>{teacherAction.reasoning}</div>
          )}
        </div>
      )}
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
    marginBottom: 8,
  },
  row: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 6,
  },
  label: {
    fontSize: 12,
    color: "#94a3b8",
    width: 80,
  },
  barBg: {
    flex: 1,
    height: 8,
    background: "#0f172a",
    borderRadius: 4,
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: 4,
    transition: "width 0.4s ease",
  },
  value: {
    fontSize: 11,
    color: "#cbd5e1",
    width: 36,
    textAlign: "right",
    fontFamily: "monospace",
  },
  summaryRow: {
    display: "flex",
    gap: 12,
    marginBottom: 8,
    padding: "5px 8px",
    background: "#0f172a",
    borderRadius: 5,
  },
  v2InfoRow: {
    display: "flex",
    gap: 12,
    marginBottom: 8,
    padding: "4px 8px",
    background: "#1e3a5f33",
    borderRadius: 5,
    borderLeft: "2px solid #60a5fa",
  },
  summaryItem: {
    fontSize: 11,
    color: "#94a3b8",
  },
  noFocus: {
    fontSize: 11,
    color: "#475569",
    padding: "6px 0",
    textAlign: "center",
  },
  actionBox: {
    marginTop: 6,
    padding: "6px 8px",
    background: "#0f172a",
    borderRadius: 5,
    borderLeft: "2px solid #334155",
  },
  actionTag: {
    fontSize: 10,
    padding: "2px 6px",
    borderRadius: 4,
    fontWeight: 700,
    textTransform: "uppercase",
  },
  actionStudent: {
    fontSize: 10,
    color: "#64748b",
    fontFamily: "monospace",
  },
  reasoning: {
    fontSize: 10,
    color: "#64748b",
    lineHeight: 1.4,
    marginTop: 2,
  },
};
