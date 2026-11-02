# Sourced by the shell scripts under code/: the three root directories and how to reach conda.
#
#   BOOK_DATA     run output, prepared inputs, vendored files, caches   default <repository>/work
#   BOOK_DATASETS the corpora, read in place                             default $BOOK_DATA/datasets
#   BOOK_MODELS   pre-trained model directories                          default $BOOK_DATA/models
#   HF_HOME       the Hugging Face cache                                 default $BOOK_DATA/cache/huggingface
#
# Conda: $CONDA_BASE if set, else `conda info --base` (conda on PATH, or $CONDA_EXE), else the first of the
# usual install directories under $HOME that holds etc/profile.d/conda.sh; otherwise the functions below stop
# with a message.

_roots_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BOOK_REPO=$(cd "$_roots_here/../.." && pwd)
export BOOK_DATA=${BOOK_DATA:-$BOOK_REPO/work}
export BOOK_DATASETS=${BOOK_DATASETS:-$BOOK_DATA/datasets}
export BOOK_MODELS=${BOOK_MODELS:-$BOOK_DATA/models}
export HF_HOME=${HF_HOME:-$BOOK_DATA/cache/huggingface}   # caches go under $BOOK_DATA/cache

book_conda_base() {  # prints the conda installation directory, or fails
    if [[ -n "${CONDA_BASE:-}" ]]; then
        echo "$CONDA_BASE"
    elif command -v "${CONDA_EXE:-conda}" >/dev/null 2>&1; then
        "${CONDA_EXE:-conda}" info --base
    else
        local d home=${HOME:-}
        if [[ -n "$home" ]]; then
            for d in "$home/miniconda3" "$home/miniforge3" "$home/anaconda3" "$home/mambaforge"; do
                if [[ -f "$d/etc/profile.d/conda.sh" ]]; then echo "$d"; return 0; fi
            done
        fi
        echo "conda not found: put conda on PATH or set CONDA_BASE to the conda installation directory" >&2
        return 1
    fi
}

book_conda_init() {  # make the `conda` shell function available in this shell, or exit
    local base
    base=$(book_conda_base) || exit 1
    # shellcheck disable=SC1091
    source "$base/etc/profile.d/conda.sh" || { echo "cannot source $base/etc/profile.d/conda.sh" >&2; exit 1; }
}

book_activate() {  # book_activate ENV: activate a conda env in this shell, or exit
    book_conda_init
    conda activate "$1" || { echo "conda activate $1 failed" >&2; exit 1; }
}

book_new_env() {  # book_new_env NAME...: exit if any of these conda envs exists; these scripts never replace one
    local e
    for e in "$@"; do
        if conda env list | awk '{print $1}' | grep -qx -- "$e"; then
            echo "conda env $e already exists; this script creates it only when it is absent (remove or rename" \
                 "the existing env yourself if it is yours)" >&2
            exit 1
        fi
    done
}

book_lock_python() {  # book_lock_python FILE: the exact Python version a lock or conda explicit file records
    local v
    v=$(grep -m1 -oE '^# python==[0-9.]+' "$1" | cut -d= -f3)
    [[ -n "$v" ]] || v=$(grep -m1 -oE '/python-[0-9]+\.[0-9]+\.[0-9]+-' "$1" | sed -E 's#/python-(.*)-#\1#')
    [[ -n "$v" ]] || { echo "no Python version recorded in $1" >&2; exit 1; }
    echo "$v"
}

book_python() {  # book_python ENV: prints the path of that env's python, or exits
    local base py
    base=$(book_conda_base) || exit 1
    py="$base/envs/$1/bin/python"
    [[ -x "$py" ]] || { echo "no python in conda env $1 ($py)" >&2; exit 1; }
    echo "$py"
}
