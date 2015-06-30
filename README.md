# fann-rs

[![Build Status](https://travis-ci.org/afck/fann-rs.svg?branch=master)](https://travis-ci.org/afck/fann-rs)
[![Crates.io](https://img.shields.io/crates/v/fann.svg?style=flat-square)](https://crates.io/crates/fann)

[Rust](http://www.rust-lang.org/) wrapper for the
[Fast Artificial Neural Network](http://leenissen.dk/fann/wp/) (FANN) library. This crate provides a
safe interface to FANN on top of the
[low-level bindings fann-sys-rs](https://github.com/afck/fann-sys-rs).

[Documentation](https://afck.github.io/docs/fann-rs/fann)


## Usage

Add `fann` and `libc` to the list of dependencies in your `Cargo.toml`:

```toml
[dependencies]
fann = "*"
libc = "*"
```

and this to your crate root:

```rust
extern crate fann;
extern crate libc;
```

Usage examples are included in the [Documentation](https://afck.github.io/docs/fann-rs/fann).
