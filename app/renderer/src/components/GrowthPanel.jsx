import React from "react";

function trendArrow(values) {
  if (!values || values.length < 2) return "→";
  const last = values[values.length - 1];
  const prev = values[values.length - 2];
  const delta = last - prev;
  if (delta > 0.02) return "↑";
  if (delta < -0.02) return "↓";
  return "→";
}

function trendColor(arrow) {
  if (arrow === "↑") return "#4ade80";
  if (arrow === "↓") return "#ef4444";
  return "#94a3b8";
}

function MetricRow({ label, value, history, format }) {
  const arrow = trendArrow(history);
  const color = trendColor(arrow);
  const display = value != null ? format(value) : "—";

  return (
    <div style={metricRowStyle}>
      <span style={{ fontSize: 11, color: "#94a3b8", width: 90 }}>{label}</span>
      <div style={{ flex: 1, height: 6, background: "#0f172a", borderRadius: 3, overflow: "hidden", margin: "0 8px" }}>
        <div
          style={{
            height: "100%",
            width: value != null ? `${Math.round(value * 100)}%` : "0%",
            background: "#3b82f6",
            borderRadius: 3,
            transition: "width 0.4s ease",
          }}
        />
      </div>
      <span style={{ fontSize: 11, color: "#e2e8f0", width: 36, textAlign: "right", fontFamily: "monospace" }}>
        {display}
      </span>
      <span style={{ fontSize: 13, color, width: 16, textAlign: "right", marginLeft: 4 }}>
        {arrow}
      </span>
    </div>
  );
}

export default function GrowthPanel({ growthData }) {
  if (!growthData) {
    return (
      <div style={styles.container}>
        <div style={styles.title}>Agent Growth</div>
        <div style={styles.empty}>수업 완료 후 성장 데이터가 표시됩니다</div>
      </div>
    );
  }

  const {
    totalClasses = 0,
    sensitivity,
    specificity,
    f1,
    ppv,
    history = {},
  } = growthData;

  return (
    <div style={styles.container}>
      <div style={styles.headerRow}>
        <span style={styles.title}>Agent Growth</span>
        <span style={styles.classBadge}>
          {totalClasses} 수업 완료
        </span>
      </div>

      <MetricRow
        label="민감도 (Recall)"
        value={sensitivity}
        history={history.sensitivity}
        format={(v) => `${(v * 100).toFixed(0)}%`}
      />
      <MetricRow
        label="정밀도 (PPV)"
        value={ppv}
        history={history.ppv}
        format={(v) => `${(v * 100).toFixed(0)}%`}
      />
      <MetricRow
        label="특이도"
        value={specificity}
        history={history.specificity}
        format={(v) => `${(v * 100).toFixed(0)}%`}
      />
      <MetricRow
        label="F1 Score"
        value={f1}
        history={history.f1}
        format={(v) => v.toFixed(3)}
      />

      {totalClasses >= 2 && (
        <div style={styles.trendNote}>
          {(() => {
            const sensArrow = trendArrow(history.sensitivity);
            const ppvArrow = trendArrow(history.ppv);
            if (sensArrow === "↑" && ppvArrow === "↑") return "전반적으로 향상 중입니다.";
            if (sensArrow === "↓" || ppvArrow === "↓") return "일부 지표가 하락했습니다.";
            return "성능이 안정적입니다.";
          })()}
        </div>
      )}
    </div>
  );
}

const metricRowStyle = {
  display: "flex",
  alignItems: "center",
  marginBottom: 6,
};

const styles = {
  container: {
    background: "#1e293b",
    borderRadius: 8,
    padding: 12,
    border: "1px solid #334155",
  },
  headerRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 10,
  },
  title: {
    fontSize: 12,
    color: "#64748b",
    textTransform: "uppercase",
    letterSpacing: 1,
    flex: 1,
  },
  classBadge: {
    fontSize: 10,
    color: "#60a5fa",
    background: "#1e3a5f55",
    padding: "2px 6px",
    borderRadius: 4,
    fontWeight: 700,
  },
  empty: {
    fontSize: 11,
    color: "#475569",
    textAlign: "center",
    padding: "8px 0",
  },
  trendNote: {
    marginTop: 4,
    fontSize: 10,
    color: "#64748b",
    borderTop: "1px solid #334155",
    paddingTop: 6,
  },
};
