import sys
import os
import subprocess

def run_unified_verification():
    print("*" * 72)
    print("     TALENT TRACK AI — UNIFIED SYSTEM-WIDE VERIFICATION SUITE       ")
    print("             Executing Milestones: Week 1 through Week 6            ")
    print("*" * 72)

    backend_dir = os.path.abspath(os.path.dirname(__file__))
    
    test_scripts = [
        ("Week 1: Foundation & Backend Architecture", "test_week1.py"),
        ("Week 2: Resume Parsing & NLP Extraction Engine", "test_week2.py"),
        ("Week 3: ML Resume Matching & ATS Composite Scoring", "test_week3.py"),
        ("Week 4: AI Recommendations, Layout & Learning Roadmap", "test_week4.py"),
        ("Week 5: Web Frontend Dashboard & API Contract", "test_week5.py"),
        ("Week 6: Deep Learning Semantic Extension & Postman Suite", "test_week6.py"),
    ]

    passed_count = 0
    total_count = len(test_scripts)

    for idx, (title, script_name) in enumerate(test_scripts, 1):
        script_path = os.path.join(backend_dir, script_name)
        print(f"\n[RUNNING MILESTONE {idx}/{total_count}] {title} ({script_name})...")
        
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  >>> SUCCESS: {title} PASSED!")
            passed_count += 1
        else:
            print(f"  >>> FAILED: {title} encountered errors.")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            sys.exit(1)

    print("\n" + "=" * 72)
    print(f"  [SUCCESS] ALL {passed_count}/{total_count} MILESTONES (WEEKS 1 TO 6) PASSED 100% SUCCESSFULLY!")
    print("  TALENT TRACK AI IS FULLY OPERATIONAL, VERIFIED, AND READY FOR USE!")
    print("=" * 72)

if __name__ == "__main__":
    run_unified_verification()
