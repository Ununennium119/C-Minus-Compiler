import subprocess
from shutil import copyfile

from compiler import Compiler


def main():
    all_passed = True
    for test_folder_name in [f"T{i}" for i in range(1, 16)]:
        print("---------------------------------------")
        print(f"running test {test_folder_name}:")
        if test_code_generator(f"testcases\\Phase-3\\{test_folder_name}", True):
            print("Passed!")
        else:
            all_passed = False
    print("---------------------------------------")
    if all_passed:
        print("All tests passed!")

    # test_code_generator("testcases\\Phase-3\\T1", True)


def test_code_generator(testcase_folder, detailed=False):
    copyfile(f"{testcase_folder}\\input.txt", "input.txt")
    compiler = Compiler()
    compiler.run()

    with open("expected.txt", "w") as expected_file:
        subprocess.call([".\\tester_windows.exe"],
                        stdout=expected_file,
                        stderr=subprocess.DEVNULL)

    with open("expected.txt", mode='r') as expected_file:
        lines = expected_file.readlines()
        lines.pop(-1)
    with open("expected.txt", mode='w') as expected_file:
        expected_file.writelines(lines)

    return check_files_equality(detailed, "expected.txt", testcase_folder)


def test_parser(testcase_folder, detailed=False) -> bool:
    copyfile(f"{testcase_folder}\\input.txt", "input.txt")
    compiler = Compiler()
    compiler.run()

    is_passed = True
    for file_name in ["parse_tree.txt", "syntax_errors.txt"]:
        is_passed &= check_files_equality(detailed, file_name, testcase_folder)
    return is_passed


def check_files_equality(detailed: bool, file_name: str, testcase_folder: str) -> bool:
    is_passed = True
    with open(file_name, "r") as output_file, open(f"{testcase_folder}\\{file_name}", "r") as testcase_output_file:
        output_file_lines = output_file.readlines()
        testcase_output_file_lines = testcase_output_file.readlines()
    if len(output_file_lines) != len(testcase_output_file_lines):
        is_passed = False
        print(f"lines count are not equal in {file_name}")
        if detailed:
            print(output_file_lines)
            print(testcase_output_file_lines)
    for i in range(min(len(output_file_lines), len(testcase_output_file_lines))):
        if output_file_lines[i].strip() != testcase_output_file_lines[i].strip():
            is_passed = False
            print(f"line {i + 1} is different in {file_name}")
            if detailed:
                print(output_file_lines[i])
                print(testcase_output_file_lines[i])
    return is_passed


def test_scanner(testcase_folder, detailed=False) -> bool:
    copyfile(f"{testcase_folder}\\input.txt", "input.txt")
    compiler = Compiler()
    compiler.run()

    is_passed = True
    for file_name in ["tokens.txt", "lexical_errors.txt"]:
        is_passed &= check_files_equality(detailed, file_name, testcase_folder)

    with open("symbol_table.txt", "r") as output_file, open(f"{testcase_folder}\\symbol_table.txt",
                                                            "r") as testcase_output_file:
        output_file_lines = output_file.readlines()
        testcase_output_file_lines = testcase_output_file.readlines()

    if len(output_file_lines) != len(testcase_output_file_lines):
        is_passed = False
        print(f"lines count are not equal in symbol_tables.txt")
        if detailed:
            print(output_file_lines)
            print(testcase_output_file_lines)
    output_file_symbols = {line.split()[1] for line in output_file_lines}
    testcase_output_file_symbols = {line.split()[1] for line in testcase_output_file_lines}
    if len(output_file_symbols.difference(testcase_output_file_symbols)) + len(
            testcase_output_file_symbols.difference(output_file_symbols)) != 0:
        is_passed = False
        print("symbol tables are not equal.")
        if detailed:
            print("output - testcase =", output_file_symbols.difference(testcase_output_file_symbols))
            print("testcase - output =", testcase_output_file_symbols.difference(output_file_symbols))

    return is_passed


if __name__ == '__main__':
    main()
