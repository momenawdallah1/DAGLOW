# Glow Lab for EffectCraft

Multi-scale glow (bloom) as an EffectCraft WebAssembly plug-in (Plug-in API v1).
Original code, no dependencies, no imports.

## Build

```
rustup target add wasm32-unknown-unknown
cargo test                                          # runs natively, checks the maths
cargo build --release --target wasm32-unknown-unknown
```

Output: `target/wasm32-unknown-unknown/release/glow_lab.wasm`

## Load

In EffectCraft: **Effect ▸ Load Effect Plug-in…** and pick the `.wasm`, or drop it in the
`Plug-ins` folder next to the settings. It appears as **Glow Lab** under Stylize.
Headless: `effectcraft-cli exec effect.plugins.load '{"path": "glow_lab.wasm"}'`

## Controls

| Control | What it does |
| --- | --- |
| Threshold | Brightness above which pixels start to glow |
| Softness | Width of the soft transition around the threshold |
| Intensity | Strength of the glow when added back |
| Radius | Blur spread; scales with preview resolution |
| Falloff | Lower keeps the glow tight, higher adds wide halos |
| Tint | Colour multiplied into the glow |
| Glow Only | Output the glow alone, without the source |

Notes: the desktop and CLI builds load `.wasm` plug-ins; the web build does not.
Each render has a fuel limit. If a very large frame leaves the image unchanged,
that is the limit: lower the preview resolution or tell me and I'll make it lighter.
