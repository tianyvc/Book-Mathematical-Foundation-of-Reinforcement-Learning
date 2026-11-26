"""用现有检查逻辑运行参考实现，而不覆盖 practice 中的作答。"""
import argparse
import importlib
from practice import check as checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("week")
    argument = parser.parse_args()
    weeks = list(checks.CHECKS) if argument.week == "all" else [int(argument.week)]
    for week in weeks:
        module = importlib.import_module(f"practice_reference.week{week:02}")
        print(f"Week {week:02}")
        checks.CHECKS[week](module)
        print("  PASS")
    print("All requested reference checks passed.")


if __name__ == "__main__":
    main()
