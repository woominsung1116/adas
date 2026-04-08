import React, { useState, useEffect, useRef, useCallback } from "react";
import ClassroomView from "./components/ClassroomView";
import StatePanel from "./components/StatePanel";
import ChatLog from "./components/ChatLog";
import ControlPanel from "./components/ControlPanel";
import StudentGrid from "./components/StudentGrid";
import GrowthPanel from "./components/GrowthPanel";

const WS_URL = window.adas?.backendUrl || "ws://localhost:8000/ws";

export default function App() {
  const [connected, setConnected] = useState(false);
  const [simState, setSimState] = useState(null);
  const [events, setEvents] = useState([]);
  const [profiles, setProfiles] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [running, setRunning] = useState(false);

  // Mode
  const [mode, setMode] = useState("classic");

  // Multi-mode state (shared by multi + v2)
  const [students, setStudents] = useState([]);
  const [classId, setClassId] = useState(null);
  const [latestTurn, setLatestTurn] = useState(null);
  const [teacherAction, setTeacherAction] = useState(null);
  const [identifiedCount, setIdentifiedCount] = useState(0);
  const [managedCount, setManagedCount] = useState(0);
  const [totalAdhd, setTotalAdhd] = useState(0);
  const [growthData, setGrowthData] = useState(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [activeScenario, setActiveScenario] = useState(null);

  // V2-specific state
  const [v2Day, setV2Day] = useState(1);
  const [v2Period, setV2Period] = useState(1);
  const [v2Subject, setV2Subject] = useState("");
  const [v2Location, setV2Location] = useState("classroom");
  const [v2MaxTurns, setV2MaxTurns] = useState(950);
  const [v2Archetype, setV2Archetype] = useState("");

  const wsRef = useRef(null);

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      setTimeout(connect, 2000);
    };
    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);

      switch (data.type) {
        case "init":
          setProfiles(data.profiles || []);
          setScenarios(data.scenarios || []);
          break;

        // ---- Classic mode ----
        case "step":
          setSimState(data.state);
          setEvents((prev) => [...prev, data]);
          break;

        case "session_end":
          setRunning(false);
          setEvents((prev) => [...prev, { ...data, type: "end" }]);
          break;

        // ---- Multi mode ----
        case "new_class":
          setClassId(data.class_id);
          setTotalAdhd(data.n_adhd || 0);
          setManagedCount(0);
          setIdentifiedCount(0);
          setTeacherAction(null);
          setLatestTurn(null);
          setEvents((prev) => [...prev, data]);
          // V2 fields
          if (data.max_turns) setV2MaxTurns(data.max_turns);
          if (data.archetype) setV2Archetype(data.archetype);
          setV2Day(1);
          setV2Period(1);
          setV2Subject("");
          setV2Location("classroom");
          break;

        case "turn": {
          const studentList = data.students || [];
          setStudents(studentList);
          setTeacherAction(data.teacher_action || null);
          setManagedCount(data.managed_count ?? 0);
          setTotalAdhd(data.total_adhd ?? 0);
          setIdentifiedCount(studentList.filter((s) => s.is_identified).length);
          setLatestTurn(data);
          setEvents((prev) => [...prev, data]);
          // V2 fields
          if (data.day != null) setV2Day(data.day);
          if (data.period != null) setV2Period(data.period);
          if (data.subject != null) setV2Subject(data.subject);
          if (data.location != null) setV2Location(data.location);
          break;
        }

        case "class_complete":
          setEvents((prev) => [...prev, data]);
          if (data.growth && Object.keys(data.growth).length > 0) {
            setGrowthData((prev) => {
              const g = data.growth;
              const history = prev?.history || {};
              return {
                totalClasses: g.total_classes,
                sensitivity: g.sensitivity,
                specificity: g.specificity,
                f1: g.f1,
                ppv: g.ppv,
                auprc: g.auprc ?? null,
                macro_f1: g.macro_f1 ?? null,
                history: {
                  sensitivity: [...(history.sensitivity || []), g.sensitivity],
                  specificity: [...(history.specificity || []), g.specificity],
                  f1: [...(history.f1 || []), g.f1],
                  ppv: [...(history.ppv || []), g.ppv],
                  auprc: [...(history.auprc || []), g.auprc ?? null],
                  macro_f1: [...(history.macro_f1 || []), g.macro_f1 ?? null],
                },
              };
            });
          }
          // V2 class_complete may have metrics directly (no growth wrapper)
          if (mode === "v2" && !data.growth && data.identified_count != null) {
            setRunning(false);
          }
          break;

        default:
          break;
      }
    };
  }, []);

  useEffect(() => {
    if (window.adas?.onBackendReady) {
      window.adas.onBackendReady(() => connect());
      // Fallback: if backend-ready never fires (e.g. port conflict), try connecting after 3s
      const fallback = setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState > 1) connect();
      }, 3000);
      return () => { clearTimeout(fallback); wsRef.current?.close(); };
    } else {
      connect();
      return () => wsRef.current?.close();
    }
  }, [connect]);

  const send = (action) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(action));
    }
  };

  const startSession = (profile, scenario) => {
    setEvents([]);
    setSimState(null);
    setStudents([]);
    setClassId(null);
    setLatestTurn(null);
    setTeacherAction(null);
    setManagedCount(0);
    setIdentifiedCount(0);
    setTotalAdhd(0);
    setPaused(false);
    setRunning(true);
    setActiveScenario(scenario || null);

    if (mode === "v2") {
      send({ type: "start_session", mode: "v2", n_students: 20 });
    } else if (mode === "multi") {
      send({ type: "start_session", mode: "multi", n_students: 20, adhd_prevalence: 0.09 });
    } else {
      send({ type: "start_session", mode: "classic", profile, scenario });
    }
  };

  const handlePause = () => {
    setPaused(true);
    send({ type: "pause" });
  };

  const handleResume = () => {
    setPaused(false);
    send({ type: "resume" });
  };

  const handleSpeedChange = (val) => {
    setSpeed(val);
    send({ type: "speed", delay: val });
  };

  const handleModeChange = (newMode) => {
    if (running) return;
    setMode(newMode);
    setEvents([]);
    setSimState(null);
    setStudents([]);
    setClassId(null);
    setLatestTurn(null);
    setTeacherAction(null);
    setV2Day(1);
    setV2Period(1);
    setV2Subject("");
    setV2Location("classroom");
    setV2Archetype("");
  };

  // Focused student for StatePanel (the one teacher is acting on)
  const focusedStudent = teacherAction?.student_id
    ? students.find((s) => s.id === teacherAction.student_id) || null
    : null;

  const multiStateProps = {
    focusedStudent,
    teacherAction,
    identifiedCount,
    managedCount,
    totalAdhd,
    v2Info: mode === "v2" ? v2Info : null,
  };

  const isMulti = mode === "multi" || mode === "v2";

  const v2Info = {
    day: v2Day,
    period: v2Period,
    subject: v2Subject,
    location: v2Location,
    maxTurns: v2MaxTurns,
    archetype: v2Archetype,
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>ADAS</h1>
        <span style={styles.subtitle}>ADHD Classroom Behavioral Simulation</span>
        {isMulti && classId != null && (
          <span style={styles.classBadge}>Class #{classId}</span>
        )}
        {mode === "v2" && running && (
          <span style={styles.classBadge}>
            Day {v2Day} · {v2Period}교시 · {v2Subject || "—"} · {v2Location}
          </span>
        )}
        <span style={{ ...styles.status, color: connected ? "#4ade80" : "#f87171" }}>
          {connected ? "Connected" : "Connecting..."}
        </span>
      </header>

      <div style={styles.main}>
        {/* Left: classroom view + (multi) student grid */}
        <div style={styles.left}>
          <ClassroomView
            state={simState}
            events={events}
            mode={mode}
            multiTurnData={latestTurn}
            scenario={activeScenario}
            v2Info={v2Info}
          />

          {isMulti && (
            <div style={styles.gridWrapper}>
              <StudentGrid
                students={students}
                classId={classId}
                targetStudentId={teacherAction?.student_id}
                managedCount={managedCount}
                totalAdhd={totalAdhd}
                mode={mode}
                v2Info={v2Info}
              />
            </div>
          )}
        </div>

        {/* Right: controls + state + log (+ growth for multi) */}
        <div style={styles.right}>
          <ControlPanel
            profiles={profiles}
            scenarios={scenarios}
            running={running}
            onStart={startSession}
            mode={mode}
            onModeChange={handleModeChange}
            paused={paused}
            onPause={handlePause}
            onResume={handleResume}
            speed={speed}
            onSpeedChange={handleSpeedChange}
            classId={classId}
            managedCount={managedCount}
            totalAdhd={totalAdhd}
          />

          <StatePanel
            state={simState}
            mode={mode}
            multiState={multiStateProps}
          />

          {isMulti && (
            <GrowthPanel growthData={growthData} mode={mode} />
          )}

          <ChatLog events={events} mode={mode} />
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    background: "#0f172a",
    color: "#e2e8f0",
    fontFamily: "'Pretendard', -apple-system, sans-serif",
  },
  header: {
    display: "flex",
    alignItems: "center",
    gap: 16,
    padding: "12px 24px",
    borderBottom: "1px solid #1e293b",
    background: "#1e293b",
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: "#60a5fa",
  },
  subtitle: {
    fontSize: 13,
    color: "#94a3b8",
    flex: 1,
  },
  classBadge: {
    fontSize: 12,
    color: "#60a5fa",
    background: "#1e3a5f",
    padding: "3px 10px",
    borderRadius: 5,
    fontWeight: 700,
    fontFamily: "monospace",
  },
  status: {
    fontSize: 12,
    fontWeight: 600,
  },
  main: {
    flex: 1,
    display: "flex",
    overflow: "hidden",
  },
  left: {
    flex: 3,
    padding: 8,
    display: "flex",
    flexDirection: "column",
    gap: 8,
    overflow: "hidden",
    minWidth: 0,
  },
  gridWrapper: {
    flexShrink: 0,
    maxHeight: 180,
    overflowY: "auto",
  },
  right: {
    flex: 1,
    minWidth: 320,
    maxWidth: 380,
    display: "flex",
    flexDirection: "column",
    gap: 8,
    padding: 16,
    borderLeft: "1px solid #1e293b",
    overflowY: "auto",
  },
};
