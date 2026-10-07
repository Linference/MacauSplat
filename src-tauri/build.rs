use std::{env, fs, path::PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let endpoint_path = manifest_dir.join("../config/telemetry-endpoint.txt");
    println!("cargo:rerun-if-changed={}", endpoint_path.display());

    let endpoint = fs::read_to_string(&endpoint_path)
        .unwrap_or_else(|error| {
            panic!(
                "failed to read telemetry endpoint from {}: {error}",
                endpoint_path.display()
            )
        })
        .trim()
        .to_owned();
    // An unconfigured derivative build stays local; telemetry is optional.
    println!("cargo:rustc-env=OOOSPLAT_TELEMETRY_ENDPOINT={endpoint}");

    tauri_build::build()
}
