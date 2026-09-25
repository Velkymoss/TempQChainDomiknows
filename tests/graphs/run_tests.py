import subprocess
import sys

TEST_FILES = [
    "tests/graphs/fr/test_transitive_determin.py",
    "tests/graphs/fr/test_symmetric.py",
    "tests/graphs/fr/test_transitive_non_determin.py",
    "tests/graphs/fr/test_inverse.py",
]


def run_test(test_file: str) -> bool:
    print("=" * 70)
    print(f"Running: {test_file}")
    print("=" * 70)
    print()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "-s"],
        capture_output=False,
    )

    print()
    if result.returncode == 0:
        print(f"✅ PASSED: {test_file}")
    else:
        print(f"❌ FAILED: {test_file}")
    print()

    return result.returncode == 0


def main():
    failed = []
    passed = 0

    for test_file in TEST_FILES:
        if run_test(test_file):
            passed += 1
        else:
            failed.append(test_file)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tests: {len(TEST_FILES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")

    if failed:
        print()
        print("Failed tests:")
        for test in failed:
            print(f"  - {test}")
        sys.exit(1)
    else:
        print()
        print("All tests passed! ✅")
        sys.exit(0)


if __name__ == "__main__":
    main()
