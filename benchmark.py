# A script to automate the timings of Rust compilation benchmarks used at
# https://www.reddit.com/r/rust/comments/qgi421/doing_m1_macbook_pro_m1_max_64gb_compile/
import sys,os
import subprocess
import tempfile
import time
import json

# You need to have the wasm32-unknown-unknown target installed, use
# `rustup target install wasm32-unknown-unknown` to get it.
# also `rustup component add rust-std`

repos = [
"https://github.com/meilisearch/MeiliSearch",
"https://github.com/denoland/deno",
"https://github.com/lunatic-solutions/lunatic",
"https://github.com/sharkdp/bat",
"https://github.com/sharkdp/hyperfine",
"https://github.com/BurntSushi/ripgrep",
"https://github.com/quickwit-inc/quickwit",
"https://github.com/sharksforarms/deku",
"https://github.com/gengjiawen/monkey-rust",
"https://github.com/getzola/zola",
"https://github.com/rust-lang/www.rust-lang.org",
"https://github.com/probe-rs/probe-rs",
"https://github.com/lycheeverse/lychee",
"https://github.com/tokio-rs/axum",
"https://github.com/paritytech/cumulus",
"https://github.com/mellowagain/gitarena",
"https://github.com/rust-analyzer/rust-analyzer",
"https://github.com/EmbarkStudios/rust-gpu",
"https://github.com/bevyengine/bevy",
"https://github.com/paritytech/substrate",
"https://github.com/bschwind/sus",
"https://github.com/artichoke/artichoke/"
]


def repo_name(url):
    return url.split("/")[-1]

def clone_repo(url, tempdir):
    """Clone the repo, returning a path to the on-disk repository if successful"""
    os.chdir(tempdir)
    cp = subprocess.run(["git", "clone", "--depth=1", url+".git", repo_name(url)])
    print(" ".join(cp.args))
    if cp.returncode != 0:
        print(f"Cloning {url} into {tempdir} was unsuccessful. Output:")
        print(cp.stdout)
        print(cp.stderr)
        cp.check_returncode()

    return os.path.join(repo_name(url))

def build_repo(workdir):
    os.chdir(workdir)
    cp = subprocess.run(["cargo","build","--release"])
    cp.check_returncode()

def main():
    oldwd = os.getcwd()
    if len(sys.argv) > 1:
        workdir = sys.argv[1]
    else:
        workdir = tempfile.mkdtemp()

    print(f"Using {workdir} as working directory")

    output = {}

    for repo_url in repos:
        os.chdir(workdir)
        try:
            print(repo_url)
            repodir = clone_repo(repo_url, workdir)
        except subprocess.CalledProcessError:
            print(f"Skipping {repo_url} due to clone failure")
            continue

        try:
            start_time = time.perf_counter()
            build_repo(repodir)
            end_time = time.perf_counter()
        except subprocess.CalledProcessError:
            print(f"Could not build {repo_url} at {repodir}. Skipping")
            output[repo_url] = "error"
            continue

        output[repo_url] = end_time - start_time
 
    outpath = os.path.join(oldwd, "results.json")
    with open(outpath,"w") as outf:
        json.dump(output, outf)

    print(f"Results written to {outpath}")


if __name__ == '__main__':
    main()
