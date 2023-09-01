import os
import ast

def get_function_signatures(file_path):
    signatures = []
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"
                signatures.append(signature)
    return signatures

def write_to_txt(file_name, data):
    with open(file_name, 'w') as f:
        for key, value in data.items():
            f.write(f"File: {key}\n")
            for signature in value:
                f.write(f"  {signature}\n")
            f.write("\n")

def main():
    directory_path = ("/Users/yuvalmoscovitz/Code/SentimentAnalysisProject/src")
    output_file = "function_signatures.txt"

    if not os.path.exists(directory_path):
        print("Directory does not exist.")
        return

    function_data = {}
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                function_data[file] = get_function_signatures(file_path)

    write_to_txt(output_file, function_data)
    print(f"Function signatures have been written to {output_file}")

if __name__ == "__main__":
    main()
