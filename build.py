import os
import sys
import subprocess
import shutil

default_clean = False

def_build_type = "release"
def_fanout = 256
def_error_bound = 64
def_min_model_f = 0.5
def_max_model_f = def_fanout
def_dataset_ratio = 1
def_query_ratio = 0.1
def_epochs = 1
def_query_range_size = 256

def delete_all_contents_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def configure_cmake(build_type, fanout, error_bound, min_size_factor, max_size_factor, dataset_ratio, query_ratio, epochs):
    if build_type.lower() == "release":
        build_type = "Release"
    elif build_type.lower() == "debug":
        build_type = "Debug"
    elif build_type.lower() == "relwithdebinfo":
        build_type = "RelWithDebInfo"
    build_dir = "build"
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        os.chdir(build_dir)
    else:
        os.chdir(build_dir)
        if default_clean == True:
            print("Clean previous build")
            make_clean = [
              "make",
              "clean"
            ]
            subprocess.run(make_clean, check=True)

    # os.chdir(build_dir)

    cmake_command = [
        "cmake",
        "-DCMAKE_BUILD_TYPE=" + build_type,
        "-DFANOUT=" + str(fanout),
        "-DMAX_EPSILON=" + str(error_bound),
        "-DMODEL_NODE_MIN_SIZE_FACTOR=" + str(min_size_factor),
        "-DMODEL_NODE_MAX_SIZE_FACTOR=" + str(max_size_factor),
        "-DDATASET_RATIO=" + str(dataset_ratio),
        "-DQUERY_RATIO=" + str(query_ratio),
        "-DEPOCH=" + str(epochs),
        "-DQUERY_RANGE_SIZE=" + str(def_query_range_size),
        ".."
    ]
    subprocess.run(cmake_command, check=True)

def build_cmake():
  subprocess.run(["cmake", "--build", "."], check=True)

def clean_build():
    if os.path.exists("build"):
        delete_all_contents_in_directory("build")
        os.rmdir("build")

def main():
    build_type = sys.argv[1] if len(sys.argv) > 1 else def_build_type
    if build_type.lower() == "clean":
        clean_build()
        return
    
    fanout = sys.argv[2] if len(sys.argv) > 2 else def_fanout
    error_bound = sys.argv[3] if len(sys.argv) > 3 else def_error_bound
    min_model_f = sys.argv[4] if len(sys.argv) > 4 else def_min_model_f
    max_model_f = sys.argv[5] if len(sys.argv) > 5 else def_max_model_f
    dataset_ratio = sys.argv[6] if len(sys.argv) > 6 else def_dataset_ratio
    query_ratio = sys.argv[7] if len(sys.argv) > 7 else def_query_ratio
    epochs = sys.argv[8] if len(sys.argv) > 8 else def_epochs

    print("Building with the following parameters:")
    print("Build Type: " + build_type)
    print("Fanout: " + str(fanout))
    print("Error Bound: " + str(error_bound))
    print("Min Model Size Factor: " + str(min_model_f))
    print("Max Model Size Factor: " + str(max_model_f))
    print("Dataset Ratio: " + str(dataset_ratio))
    print("Query Ratio: " + str(query_ratio))
    print("Epochs: " + str(epochs))
    print("Query Range Size: " + str(def_query_range_size))

    configure_cmake(build_type, fanout, error_bound, min_model_f, max_model_f, dataset_ratio, query_ratio, epochs)
    build_cmake()


if __name__ == "__main__":
    main()