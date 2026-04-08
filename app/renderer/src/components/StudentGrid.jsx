import React from "react";

// Status color mapping
function getStatusColor(student) {
  if (student.is_managed) return "#a855f7"; // purple - managed
  if (student.is_identified) return "#60a5fa"; // blue - identified ADHD
  const s = student.state || {};
  const escalation = s.escalation ?? s.escalation_risk ?? 0;
  const distress = s.distress ?? s.distress_level ?? 0;
  if (escalation > 0.6 || distress > 0.7) return "#ef4444"; // red - escalating
  if (escalation > 0.35 || distress > 0.45) return "#f59e0b"; // yellow - suspicious
  return "#4ade80"; // green - normal
}

function getStatusLabel(student) {
  if (student.is_managed) return "관리됨";
  if (student.is_identified) return "ADHD";
  const s = student.state || {};
  const escalation = s.escalation ?? s.escalation_risk ?? 0;
  const distress = s.distress ?? s.distress_level ?? 0;
  if (escalation > 0.6 || distress > 0.7) return "위험";
  if (escalation > 0.35 || distress > 0.45) return "주의";
  return "정상";
}

function StudentCard({ student, isTargeted }) {
  const color = getStatusColor(student);
  const label = getStatusLabel(student);
  const compliance = Math.round((student.state?.compliance ?? 0) * 100);
  const behaviors = student.behaviors || [];

  return (
    <div
      style={{
        ...cardStyle,
        borderColor: isTargeted ? "#fbbf24" : color + "55",
        boxShadow: isTargeted ? `0 0 0 2px #fbbf24, 0 0 8px #fbbf2466` : "none",
        background: isTargeted ? "#1e2d3d" : "#0f172a",
      }}
    >
      {/* Header row: name + status dot */}
      <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
            boxShadow: isTargeted ? `0 0 5px ${color}` : "none",
          }}
        />
        <span style={{ fontSize: 11, fontWeight: 600, color: "#e2e8f0", flex: 1, minWidth: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {student.name}
        </span>
      </div>

      {/* Badge row */}
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginBottom: 3 }}>
        <span style={{ ...badgeStyle, background: color + "33", color }}>
          {label}
        </span>
        {student.is_identified && !student.is_managed && (
          <span style={{ ...badgeStyle, background: "#1e40af55", color: "#93c5fd" }}>
            ADHD
          </span>
        )}
        {student.is_managed && (
          <span style={{ ...badgeStyle, background: "#581c8755", color: "#d8b4fe" }}>
            ✓
          </span>
        )}
      </div>

      {/* Compliance bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <span style={{ fontSize: 9, color: "#64748b", width: 16, flexShrink: 0 }}>C:</span>
        <div style={{ flex: 1, height: 4, background: "#1e293b", borderRadius: 2, overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${compliance}%`, background: compliance > 60 ? "#4ade80" : compliance > 30 ? "#f59e0b" : "#ef4444", borderRadius: 2, transition: "width 0.3s" }} />
        </div>
        <span style={{ fontSize: 9, color: "#94a3b8", width: 22, textAlign: "right", fontFamily: "monospace" }}>
          {compliance}%
        </span>
      </div>

      {/* Emotional indicators */}
      {student.state && (student.state.distress > 0.4 || student.state.escalation > 0.4) && (
        <div style={{ display: "flex", gap: 3, marginTop: 2 }}>
          {student.state.distress > 0.4 && (
            <span style={{ fontSize: 8, color: "#ef4444", background: "#ef444422", padding: "0 3px", borderRadius: 2 }}>
              distress {Math.round((student.state.distress ?? 0) * 100)}%
            </span>
          )}
          {student.state.escalation > 0.4 && (
            <span style={{ fontSize: 8, color: "#f59e0b", background: "#f59e0b22", padding: "0 3px", borderRadius: 2 }}>
              esc {Math.round((student.state.escalation ?? 0) * 100)}%
            </span>
          )}
        </div>
      )}

      {/* Behaviors (tiny text) */}
      {behaviors.length > 0 && (
        <div style={{ marginTop: 3, fontSize: 9, color: "#64748b", lineHeight: 1.3, overflow: "hidden", maxHeight: 24 }}>
          {behaviors.slice(0, 2).join(", ")}
        </div>
      )}

      {/* Targeted pulse indicator */}
      {isTargeted && (
        <div style={{ position: "absolute", top: 2, right: 4, fontSize: 9, color: "#fbbf24", fontWeight: 700 }}>
          ▶
        </div>
      )}
    </div>
  );
}

export default function StudentGrid({ students, classId, targetStudentId, managedCount, totalAdhd, mode, v2Info }) {
  if (!students || students.length === 0) {
    return (
      <div style={styles.empty}>
        학생 데이터 대기 중...
      </div>
    );
  }

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <span style={styles.headerTitle}>학생 현황</span>
        {classId != null && (
          <span style={styles.classTag}>Class #{classId}</span>
        )}
        {mode === "v2" && v2Info && (
          <span style={styles.v2Tag}>
            Day {v2Info.day} · {v2Info.period}교시 · {v2Info.subject || "—"}
          </span>
        )}
        {mode === "v2" && v2Info?.archetype && (
          <span style={styles.v2Tag}>{v2Info.archetype}</span>
        )}
        {totalAdhd != null && (
          <span style={styles.stat}>
            관리: <b style={{ color: "#a855f7" }}>{managedCount ?? 0}</b>/{totalAdhd} ADHD
          </span>
        )}
      </div>

      {/* Grid */}
      <div style={styles.grid}>
        {students.map((student) => (
          <StudentCard
            key={student.id}
            student={student}
            isTargeted={student.id === targetStudentId}
          />
        ))}
      </div>
    </div>
  );
}

const cardStyle = {
  position: "relative",
  background: "#0f172a",
  border: "1px solid #334155",
  borderRadius: 6,
  padding: "5px 6px",
  cursor: "default",
  transition: "border-color 0.2s, box-shadow 0.2s",
};

const badgeStyle = {
  fontSize: 9,
  padding: "1px 4px",
  borderRadius: 3,
  fontWeight: 600,
  lineHeight: 1.4,
};

const styles = {
  container: {
    background: "#1e293b",
    borderRadius: 8,
    padding: 10,
    border: "1px solid #334155",
  },
  empty: {
    background: "#1e293b",
    borderRadius: 8,
    padding: 16,
    border: "1px solid #334155",
    color: "#64748b",
    fontSize: 12,
    textAlign: "center",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 8,
  },
  headerTitle: {
    fontSize: 11,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 1,
    fontWeight: 600,
    flex: 1,
  },
  classTag: {
    fontSize: 10,
    color: "#60a5fa",
    background: "#1e3a5f55",
    padding: "2px 6px",
    borderRadius: 4,
    fontWeight: 700,
  },
  v2Tag: {
    fontSize: 9,
    color: "#60a5fa",
    background: "#1e3a5f55",
    padding: "2px 5px",
    borderRadius: 3,
    fontWeight: 600,
  },
  stat: {
    fontSize: 10,
    color: "#94a3b8",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: 4,
  },
};
