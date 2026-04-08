import React, { useRef, useEffect } from "react";

const ACTION_COLORS = {
  observe: "#60a5fa",
  identify_adhd: "#fbbf24",
  individual_intervention: "#4ade80",
  class_instruction: "#94a3b8",
  correct: "#ef4444",
};

function ClassicEntry({ ev }) {
  return (
    <>
      <div style={styles.meta}>
        <span style={styles.turn}>T{ev.time || ev.turn || 0}</span>
        <span style={styles.action}>{ev.action || "setup"}</span>
        {ev.reward !== undefined && (
          <span style={{ ...styles.reward, color: ev.reward >= 0 ? "#4ade80" : "#ef4444" }}>
            {ev.reward >= 0 ? "+" : ""}{ev.reward?.toFixed(3)}
          </span>
        )}
      </div>
      {ev.utterance && (
        <div style={styles.teacher}>{ev.speaker}: {ev.utterance}</div>
      )}
      {ev.student_narrative && (
        <div style={styles.student}>{ev.student_narrative}</div>
      )}
      {ev.observer_note?.note && (
        <div style={styles.observer}>{ev.observer_note.note}</div>
      )}
    </>
  );
}

function MultiTurnEntry({ ev }) {
  const action = ev.teacher_action || {};
  const actionType = action.action_type || "observe";
  const color = ACTION_COLORS[actionType] || "#94a3b8";

  // Find targeted student name
  const targetStudent = ev.students?.find((s) => s.id === action.student_id);
  const targetName = targetStudent?.name || action.student_id || "";

  return (
    <>
      <div style={styles.meta}>
        <span style={styles.turn}>T{ev.turn}</span>
        <span style={{ ...styles.action, color }}>
          {actionType.replace(/_/g, " ")}
        </span>
        {targetName && (
          <span style={styles.targetName}>{targetName}</span>
        )}
        {ev.reward !== undefined && (
          <span style={{ ...styles.reward, color: ev.reward >= 0 ? "#4ade80" : "#ef4444" }}>
            {ev.reward >= 0 ? "+" : ""}{ev.reward?.toFixed(3)}
          </span>
        )}
      </div>
      {action.reasoning && (
        <div style={styles.reasoning}>{action.reasoning}</div>
      )}
      {ev.identifications?.length > 0 && ev.identifications.map((id, i) => (
        <div key={i} style={styles.identBadge}>
          ADHD 식별: {ev.students?.find((s) => s.id === id.student_id)?.name || id.student_id}
          {id.confidence > 0 && ` (신뢰도 ${(id.confidence * 100).toFixed(0)}%)`}
        </div>
      ))}
      {ev.memory_summary && (
        <div style={styles.memorySummary}>{ev.memory_summary}</div>
      )}
    </>
  );
}

function NewClassEntry({ ev }) {
  return (
    <div style={styles.newClass}>
      <span style={styles.newClassBadge}>Class #{ev.class_id}</span>
      <span style={{ fontSize: 11, color: "#94a3b8" }}>
        학생 {ev.n_students}명 · ADHD {ev.n_adhd}명 · {ev.scenario || ""}
      </span>
    </div>
  );
}

function ClassCompleteEntry({ ev }) {
  const g = ev.growth || {};
  return (
    <div style={styles.classComplete}>
      <div style={{ fontWeight: 700, color: "#4ade80", marginBottom: 3 }}>
        Class #{ev.class_id} 완료
      </div>
      <div style={{ fontSize: 10, color: "#94a3b8" }}>
        TP:{ev.true_positives} FP:{ev.false_positives} FN:{ev.false_negatives} · 관리:{ev.managed_count}
      </div>
      {g.sensitivity != null && (
        <div style={{ fontSize: 10, color: "#60a5fa", marginTop: 2 }}>
          민감도 {(g.sensitivity * 100).toFixed(0)}% · PPV {(g.ppv * 100).toFixed(0)}% · F1 {g.f1?.toFixed(3)}
        </div>
      )}
    </div>
  );
}

export default function ChatLog({ events, mode }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  return (
    <div style={styles.container}>
      <div style={styles.title}>Interaction Log</div>
      <div style={styles.list}>
        {events.map((ev, i) => (
          <div key={i} style={styles.entry}>
            {ev.type === "end" || ev.type === "session_end" ? (
              <div style={styles.endBadge}>
                {ev.summary?.status === "success" ? "SUCCESS" : "ENDED"}
                {ev.summary?.total_reward != null && ` - Reward: ${ev.summary.total_reward.toFixed(2)}`}
              </div>
            ) : ev.type === "new_class" ? (
              <NewClassEntry ev={ev} />
            ) : ev.type === "class_complete" ? (
              <ClassCompleteEntry ev={ev} />
            ) : ev.type === "turn" ? (
              <MultiTurnEntry ev={ev} />
            ) : (
              <ClassicEntry ev={ev} />
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

const styles = {
  container: {
    flex: 1,
    background: "#1e293b",
    borderRadius: 8,
    padding: 12,
    border: "1px solid #334155",
    display: "flex",
    flexDirection: "column",
    minHeight: 0,
  },
  title: {
    fontSize: 12,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 1,
    marginBottom: 8,
  },
  list: {
    flex: 1,
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 8,
  },
  entry: {
    padding: 8,
    background: "#0f172a",
    borderRadius: 6,
    fontSize: 12,
  },
  meta: {
    display: "flex",
    gap: 8,
    alignItems: "center",
    marginBottom: 4,
  },
  turn: {
    color: "#64748b",
    fontFamily: "monospace",
    fontSize: 11,
  },
  action: {
    color: "#60a5fa",
    fontWeight: 600,
    fontSize: 11,
  },
  targetName: {
    fontSize: 11,
    color: "#e2e8f0",
    background: "#1e293b",
    padding: "1px 5px",
    borderRadius: 3,
  },
  reward: {
    fontFamily: "monospace",
    fontSize: 11,
    marginLeft: "auto",
  },
  teacher: {
    color: "#93c5fd",
    lineHeight: 1.4,
    marginBottom: 2,
  },
  student: {
    color: "#fde68a",
    lineHeight: 1.4,
    fontStyle: "italic",
    marginBottom: 2,
  },
  observer: {
    color: "#6b7280",
    fontSize: 11,
    lineHeight: 1.3,
  },
  reasoning: {
    color: "#64748b",
    fontSize: 10,
    lineHeight: 1.4,
    marginBottom: 2,
  },
  identBadge: {
    color: "#fbbf24",
    fontSize: 10,
    background: "#451a0355",
    padding: "2px 6px",
    borderRadius: 3,
    marginTop: 2,
    fontWeight: 600,
  },
  memorySummary: {
    color: "#475569",
    fontSize: 9,
    marginTop: 2,
    fontFamily: "monospace",
  },
  newClass: {
    display: "flex",
    alignItems: "center",
    gap: 8,
  },
  newClassBadge: {
    fontSize: 11,
    color: "#60a5fa",
    background: "#1e3a5f",
    padding: "2px 7px",
    borderRadius: 4,
    fontWeight: 700,
  },
  classComplete: {
    padding: 2,
  },
  endBadge: {
    textAlign: "center",
    color: "#fbbf24",
    fontWeight: 700,
    padding: 4,
  },
};
