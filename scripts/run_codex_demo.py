#!/usr/bin/env python3
"""Quick demo: run 1 class with Codex CLI for ALL agents (teacher + students).

Usage:
    .venv/bin/python scripts/run_codex_demo.py
"""
import sys
import os
import json
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.llm.codex_cli_backend import CodexCLIBackend
from src.llm.student_llm import StudentLLM, StudentContext, StudentResponse
from src.simulation.multi_student_env import MultiStudentClassroom, TeacherAction
from src.simulation.teacher_memory import TeacherMemory


def main():
    print("=== ADAS Codex Demo: All Agents via Codex CLI ===\n")

    # Setup
    backend = CodexCLIBackend(
        command="codex",
        cache_enabled=True,
        cache_dir=".cache/codex_demo",
        timeout=120,
    )
    student_llm = StudentLLM(backend, cache_dir=".cache/codex_student")
    memory = TeacherMemory()

    # Small class for demo (10 students, faster)
    classroom = MultiStudentClassroom(
        n_students=10,
        adhd_prevalence=(0.1, 0.2),  # 1-2 ADHD students
        seed=42,
    )
    obs = classroom.reset()

    n_adhd = sum(1 for s in classroom.students if s.is_adhd)
    print(f"Class: {len(classroom.students)} students, {n_adhd} ADHD\n")

    # Print student roster (debug)
    for s in classroom.students:
        label = f"ADHD({s.adhd_subtype},{s.severity})" if s.is_adhd else "Normal"
        print(f"  {s.student_id}: {label}, gender={s.gender}, age={s.age}")
    print()

    # Run 5 turns
    max_turns = 5
    for turn in range(1, max_turns + 1):
        print(f"--- Turn {turn} ---")
        t0 = time.time()

        # Step 1: Teacher decides action (via Codex CLI)
        # For demo, use simple rule: observe students in rotation
        target_idx = (turn - 1) % len(classroom.students)
        target = classroom.students[target_idx]
        action = TeacherAction(
            action_type="observe",
            student_id=target.student_id,
            reasoning=f"Turn {turn}: observing {target.student_id}",
        )
        print(f"  Teacher: {action.action_type} → {action.student_id}")

        # Step 2: Each student responds via Codex CLI
        context = StudentContext(
            scenario="math_class",
            teacher_action=action.action_type,
            teacher_utterance="",
            turn=turn,
            class_mood="calm",
        )

        student_responses = {}
        for student in classroom.students:
            try:
                resp = student_llm.generate_response(student, context)
                student_responses[student.student_id] = resp
                # Update student state from LLM response
                student.state = resp.state
                student.exhibited_behaviors = resp.behaviors
                print(f"    {student.student_id}: {resp.narrative[:60]}...")
                if resp.inner_thought:
                    print(f"      (내면: {resp.inner_thought[:50]}...)")
            except Exception as e:
                print(f"    {student.student_id}: ERROR - {e}")

        elapsed = time.time() - t0
        print(f"  Turn time: {elapsed:.1f}s ({elapsed/len(classroom.students):.1f}s/student)\n")

    print("=== Demo Complete ===")
    print(f"Total students processed: {len(classroom.students)} × {max_turns} turns")
    print(f"Cache dir: .cache/codex_demo, .cache/codex_student")


if __name__ == "__main__":
    main()
