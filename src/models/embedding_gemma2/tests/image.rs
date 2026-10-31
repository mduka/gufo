//! Host image preprocessing against the reference processor: the patch grid
//! and the SHA-256 of the resized patch bytes (`image_golden_test.cpp`):
//! cargo test -p gufo-embedding-gemma2 --test image

#![allow(clippy::unwrap_used)]

mod common;

use common::make_image;
use gufo_core::image::Image;
use gufo_core::json;
use gufo_embedding_gemma2::image_processor::process_image;
use sha2::{Digest, Sha256};

#[test]
fn patches_match_the_reference_processor() {
    let path = format!(
        "{}/tests/fixtures/image_golden.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture = json::parse(&std::fs::read(path).unwrap()).unwrap();
    for case in fixture.find("cases").unwrap().items() {
        let name = case.member_str("name", "");
        for specification in case.find("images").unwrap().items() {
            let patches = process_image(&make_image(specification)).unwrap();
            assert_eq!(
                (patches.rows as usize, patches.columns as usize),
                (
                    specification.member_size("rows", 0),
                    specification.member_size("columns", 0)
                ),
                "{name} patch grid"
            );
            let bytes: Vec<u8> = patches
                .pixels
                .iter()
                .map(|&pixel| (pixel * 255.0).round() as u8)
                .collect();
            let hash: String = Sha256::digest(&bytes)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            assert_eq!(
                hash,
                specification.member_str("patch_sha256", ""),
                "{name} resized patch bytes"
            );
        }
    }
}

#[test]
fn invalid_images_are_refused() {
    assert!(process_image(&Image::new(1, 1, vec![0; 3]).unwrap()).is_ok());
    assert!(process_image(&Image::new(40000, 1, vec![0; 120000]).unwrap()).is_err());
}
