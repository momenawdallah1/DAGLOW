//! Glow Lab 2: a multi-scale glow (bloom) effect for EffectCraft (Plug-in API v1).
//!
//! Written from scratch. The method is the standard bloom pipeline:
//!   1. bright-pass with a soft knee,
//!   2. a pyramid of progressively smaller, blurred copies (each level is wider on screen),
//!   3. Radius decides how many of the wide levels take part, Spread weights them,
//!   4. blend the result over the layer (Screen, Add or Lighten), tinted by Tint.
//!
//! No imports, no dependencies, deterministic float maths.

use core::cell::UnsafeCell;

const LEVELS: usize = 7;
const KERNEL: [f32; 5] = [0.227_027, 0.194_594_6, 0.121_621_6, 0.054_054, 0.016_216];
/// Flattened f64 parameters: 4 sliders/popup..., see the manifest order below.
const NPARAMS: usize = 13;
/// Overall strength at Exposure 1.0.
const BASE_GAIN: f32 = 3.0;

const MANIFEST: &str = r#"{
  "api": 1,
  "id": "com.glowlab.multiscale-glow",
  "name": "Glow Lab",
  "category": "Stylize",
  "version": "2.0.0",
  "author": "Glow Lab",
  "description": "Multi-scale glow with a pixel radius, spread, threshold and blend modes.",
  "params": [
    {"id": "radius", "name": "Radius", "type": "slider", "default": 250, "min": 0, "max": 5000, "sliderMax": 1000, "decimals": 0},
    {"id": "exposure", "name": "Exposure", "type": "slider", "default": 1.0, "min": 0, "max": 50, "sliderMax": 5, "decimals": 2},
    {"id": "threshold", "name": "Threshold", "type": "slider", "default": 50, "min": 0, "max": 100, "decimals": 0},
    {"id": "threshold_smooth", "name": "Threshold Smooth", "type": "slider", "default": 50, "min": 0, "max": 100, "decimals": 0},
    {"id": "blend_mode", "name": "Blend Mode", "type": "popup", "options": ["Screen", "Add", "Lighten"], "default": 0},
    {"id": "aspect_ratio", "name": "Aspect Ratio", "type": "slider", "default": 1.0, "min": 0.2, "max": 5, "decimals": 2},
    {"id": "spread", "name": "Spread", "type": "slider", "default": 33, "min": 0, "max": 100, "decimals": 0},
    {"id": "tint", "name": "Tint", "type": "color", "default": [1.0, 1.0, 1.0, 1.0]},
    {"id": "view", "name": "View", "type": "popup", "options": ["Final Render", "Glow Only", "Source"], "default": 0},
    {"id": "source_opacity", "name": "Source Opacity", "type": "slider", "default": 100, "min": 0, "max": 100, "decimals": 0}
  ]
}"#;

// Flattened parameter order:
//  0 radius | 1 exposure | 2 threshold | 3 threshold smooth | 4 blend mode | 5 aspect ratio
//  6 spread | 7..10 tint rgba | 11 view | 12 source opacity

// ---------------------------------------------------------------------------
// ABI exports
// ---------------------------------------------------------------------------

struct Buf(UnsafeCell<Vec<u64>>);
// Each plug-in instance is driven by one thread at a time.
unsafe impl Sync for Buf {}
static BUF: Buf = Buf(UnsafeCell::new(Vec::new()));

#[no_mangle]
pub extern "C" fn ec_api_version() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn ec_manifest_ptr() -> i32 {
    MANIFEST.as_ptr() as usize as i32
}

#[no_mangle]
pub extern "C" fn ec_manifest_len() -> i32 {
    MANIFEST.len() as i32
}

#[no_mangle]
pub extern "C" fn ec_alloc(bytes: i32) -> i32 {
    let words = ((bytes.max(0) as usize) + 7) / 8;
    let v = unsafe { &mut *BUF.0.get() };
    if v.len() < words {
        *v = vec![0u64; words.max(1)];
    }
    v.as_mut_ptr() as usize as i32
}

#[no_mangle]
pub extern "C" fn ec_render(
    pixels: i32,
    width: i32,
    height: i32,
    params: i32,
    nparams: i32,
    _time: f64,
    scale: f64,
) -> i32 {
    if width <= 0 || height <= 0 || nparams < 0 || pixels <= 0 || params <= 0 {
        return 2;
    }
    let (w, h) = (width as usize, height as usize);
    let px = unsafe { core::slice::from_raw_parts_mut(pixels as u32 as usize as *mut f32, w * h * 4) };
    let p = unsafe { core::slice::from_raw_parts(params as u32 as usize as *const f64, nparams as usize) };
    match render(px, w, h, p, scale) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

// ---------------------------------------------------------------------------
// Effect
// ---------------------------------------------------------------------------

fn fin(v: f64, lo: f32, hi: f32) -> f32 {
    let x = v as f32;
    if x.is_finite() {
        x.clamp(lo, hi)
    } else {
        lo
    }
}

/// Soft-knee bright pass on a premultiplied pixel. Scales all four channels by the
/// same factor so the result stays a valid premultiplied colour.
fn bright(px: &[f32], thr: f32, knee: f32) -> [f32; 4] {
    let a = px[3];
    if a <= 1e-5 {
        return [0.0; 4];
    }
    let m = px[0].max(px[1]).max(px[2]) / a;
    let mut s = (m - thr + knee).clamp(0.0, 2.0 * knee);
    s = s * s / (4.0 * knee + 1e-4);
    let k = s.max(m - thr).max(0.0) / m.max(1e-4);
    [px[0] * k, px[1] * k, px[2] * k, a * k]
}

/// 2x2 box downsample; `get` supplies the source pixel at (x, y).
fn down<F: Fn(usize, usize) -> [f32; 4]>(w: usize, h: usize, get: F) -> (Vec<f32>, usize, usize) {
    let dw = (w + 1) / 2;
    let dh = (h + 1) / 2;
    let mut out = vec![0.0f32; dw * dh * 4];
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = [0.0f32; 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let v = get(sx, sy);
                    for c in 0..4 {
                        acc[c] += v[c];
                    }
                }
            }
            let o = (y * dw + x) * 4;
            for c in 0..4 {
                out[o + c] = acc[c] * 0.25;
            }
        }
    }
    (out, dw, dh)
}

/// Separable 9-tap Gaussian pass with a fractional tap spacing (`step` pixels).
fn blur_axis(src: &[f32], w: usize, h: usize, step: f32, horizontal: bool) -> Vec<f32> {
    if step < 0.01 {
        return src.to_vec();
    }
    let mut out = vec![0.0f32; src.len()];
    let (lines, len, line_stride, pos_stride) = if horizontal {
        (h, w, w * 4, 4)
    } else {
        (w, h, 4, w * 4)
    };
    let max_pos = (len - 1) as f32;
    for l in 0..lines {
        let base = l * line_stride;
        for p in 0..len {
            let mut acc = [0.0f32; 4];
            for c in 0..4 {
                acc[c] = src[base + p * pos_stride + c] * KERNEL[0];
            }
            for k in 1..5 {
                let off = k as f32 * step;
                for sign in [1.0f32, -1.0f32] {
                    let pos = (p as f32 + sign * off).clamp(0.0, max_pos);
                    let i0 = pos as usize;
                    let i1 = (i0 + 1).min(len - 1);
                    let f = pos - i0 as f32;
                    for c in 0..4 {
                        let a = src[base + i0 * pos_stride + c];
                        let b = src[base + i1 * pos_stride + c];
                        acc[c] += (a + (b - a) * f) * KERNEL[k];
                    }
                }
            }
            for c in 0..4 {
                out[base + p * pos_stride + c] = acc[c];
            }
        }
    }
    out
}

/// Bilinearly samples `src` (sw x sh) at every pixel of a dw x dh grid and calls
/// `f(offset_of_pixel_in_dst, value)`.
fn upsample_map<F: FnMut(usize, [f32; 4])>(dw: usize, dh: usize, src: &[f32], sw: usize, sh: usize, mut f: F) {
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * sh as f32 / dh as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * sw as f32 / dw as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = fx - x0 as f32;
            let i00 = (y0 * sw + x0) * 4;
            let i10 = (y0 * sw + x1) * 4;
            let i01 = (y1 * sw + x0) * 4;
            let i11 = (y1 * sw + x1) * 4;
            let mut v = [0.0f32; 4];
            for c in 0..4 {
                let top = src[i00 + c] + (src[i10 + c] - src[i00 + c]) * tx;
                let bot = src[i01 + c] + (src[i11 + c] - src[i01 + c]) * tx;
                v[c] = top + (bot - top) * ty;
            }
            f((y * dw + x) * 4, v);
        }
    }
}

fn scaled(v: &[f32], k: f32) -> Vec<f32> {
    v.iter().map(|x| x * k).collect()
}

fn mix(mode: u8, s: f32, g: f32) -> f32 {
    match mode {
        0 => s + g - s * g, // Screen
        1 => s + g,         // Add
        _ => s.max(g),      // Lighten
    }
}

/// Core of the effect, separated from the raw-pointer ABI so it can be unit-tested.
pub fn render(px: &mut [f32], w: usize, h: usize, p: &[f64], scale: f64) -> Result<(), i32> {
    if w == 0 || h == 0 || px.len() < w * h * 4 || p.len() < NPARAMS {
        return Err(1);
    }
    let px = &mut px[..w * h * 4];

    let radius = fin(p[0], 0.0, 5000.0);
    let exposure = fin(p[1], 0.0, 50.0);
    let thr = fin(p[2], 0.0, 100.0) / 100.0;
    let knee = fin(p[3], 0.0, 100.0) / 100.0 * 0.3;
    let blend = (p[4] as i64).clamp(0, 2) as u8;
    let aspect = fin(p[5], 0.2, 5.0);
    let spread = fin(p[6], 0.0, 100.0);
    let tint = [fin(p[7], 0.0, 10.0), fin(p[8], 0.0, 10.0), fin(p[9], 0.0, 10.0)];
    let view = (p[11] as i64).clamp(0, 2);
    let opacity = fin(p[12], 0.0, 100.0) / 100.0;
    let sc = if scale.is_finite() && scale > 0.0 { scale as f32 } else { 1.0 };

    // View: Source only needs the opacity.
    if view == 2 {
        for v in px.iter_mut() {
            *v *= opacity;
        }
        return Ok(());
    }

    // Aspect Ratio stretches the blur: > 1 widens horizontally, < 1 vertically.
    let step_h = aspect.sqrt();
    let step_v = 1.0 / aspect.sqrt();

    // 1 + 2: bright-pass folded into the first downsample, then blur each level.
    let (mut cur, mut cw, mut ch) = down(w, h, |x, y| {
        let i = (y * w + x) * 4;
        bright(&px[i..i + 4], thr, knee)
    });
    let mut levels: Vec<Vec<f32>> = Vec::with_capacity(LEVELS);
    let mut dims: Vec<(usize, usize)> = Vec::with_capacity(LEVELS);
    for i in 0..LEVELS {
        let t = blur_axis(&cur, cw, ch, step_h, true);
        let b = blur_axis(&t, cw, ch, step_v, false);
        let next = if i + 1 < LEVELS {
            Some(down(cw, ch, |x, y| {
                let j = (y * cw + x) * 4;
                [b[j], b[j + 1], b[j + 2], b[j + 3]]
            }))
        } else {
            None
        };
        levels.push(b);
        dims.push((cw, ch));
        if let Some((n, nw, nh)) = next {
            cur = n;
            cw = nw;
            ch = nh;
        }
    }

    // 3: Spread weights the levels; Radius gates the wide levels in or out.
    // Level i is d = 2^(i+1) times smaller than the frame and blurs to about 2.07 * d pixels.
    let falloff = 0.45 + 0.01 * spread;
    let mut ws = [0.0f32; LEVELS];
    let mut pw = 1.0f32;
    let mut sum = 0.0f32;
    for i in 0..LEVELS {
        ws[i] = pw;
        sum += pw;
        pw *= falloff;
    }
    let reach = (radius * sc).max(1.0) * 0.6;
    let mut d = 2.0f32;
    for i in 0..LEVELS {
        let r = 2.07 * d / reach;
        let r2 = r * r;
        let gate = 1.0 / (1.0 + r2 * r2);
        ws[i] = ws[i] / sum * BASE_GAIN * gate;
        d *= 2.0;
    }

    // Fold the levels from coarse to fine into one glow image at the finest level.
    let mut acc = scaled(&levels[LEVELS - 1], ws[LEVELS - 1]);
    for i in (0..LEVELS - 1).rev() {
        let mut here = scaled(&levels[i], ws[i]);
        upsample_map(dims[i].0, dims[i].1, &acc, dims[i + 1].0, dims[i + 1].1, |o, v| {
            for c in 0..4 {
                here[o + c] += v[c];
            }
        });
        acc = here;
    }

    // 4: Source Opacity applies to the visible source, not to what feeds the glow.
    if opacity < 1.0 {
        for v in px.iter_mut() {
            *v *= opacity;
        }
    }
    let gains = [tint[0] * exposure, tint[1] * exposure, tint[2] * exposure, exposure];
    let glow_only = view == 1;
    upsample_map(w, h, &acc, dims[0].0, dims[0].1, |o, g| {
        for c in 0..4 {
            let gl = g[c] * gains[c];
            px[o + c] = if glow_only { gl } else { mix(blend, px[o + c], gl) };
        }
        if px[o + 3] > 1.0 {
            px[o + 3] = 1.0;
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // radius 20, exposure 1, threshold 50, smooth 50, Screen, aspect 1, spread 33,
    // tint white, Final Render, opacity 100
    const DEFAULTS: [f64; NPARAMS] = [20.0, 1.0, 50.0, 50.0, 0.0, 1.0, 33.0, 1.0, 1.0, 1.0, 1.0, 0.0, 100.0];

    fn dot_image(n: usize) -> Vec<f32> {
        let mut px = vec![0.0f32; n * n * 4];
        for i in 0..n * n {
            px[i * 4 + 3] = 1.0; // opaque black
        }
        let c = (n / 2 * n + n / 2) * 4;
        px[c] = 1.0;
        px[c + 1] = 1.0;
        px[c + 2] = 1.0;
        px
    }

    #[test]
    fn manifest_is_consistent() {
        assert!(MANIFEST.contains("\"api\": 1"));
        assert_eq!(MANIFEST.len() as i32, ec_manifest_len());
        assert_eq!(ec_api_version(), 1);
    }

    #[test]
    fn bright_dot_spreads_light_to_neighbours() {
        let n = 64;
        let mut px = dot_image(n);
        render(&mut px, n, n, &DEFAULTS, 1.0).unwrap();
        let at = |x: usize, y: usize| px[(y * n + x) * 4];
        assert!(at(n / 2, n / 2) >= 0.999);
        assert!(at(n / 2 + 3, n / 2) > 1e-4, "glow should reach nearby pixels");
        assert!(at(n / 2 + 3, n / 2) > at(n / 2 + 25, n / 2), "glow should fall off with distance");
    }

    #[test]
    fn larger_radius_reaches_further() {
        let n = 128;
        let mut small = dot_image(n);
        let mut large = dot_image(n);
        let mut p = DEFAULTS;
        p[0] = 8.0;
        render(&mut small, n, n, &p, 1.0).unwrap();
        p[0] = 250.0;
        render(&mut large, n, n, &p, 1.0).unwrap();
        let far = (n / 2 * n + n / 2 + 40) * 4;
        assert!(large[far] > small[far]);
    }

    #[test]
    fn zero_exposure_leaves_the_image_unchanged() {
        let n = 32;
        let original = dot_image(n);
        let mut px = original.clone();
        let mut p = DEFAULTS;
        p[1] = 0.0;
        render(&mut px, n, n, &p, 1.0).unwrap();
        assert_eq!(px, original);
    }

    #[test]
    fn dark_image_stays_dark() {
        let n = 32;
        let mut px = vec![0.0f32; n * n * 4];
        for i in 0..n * n {
            px[i * 4] = 0.2;
            px[i * 4 + 1] = 0.2;
            px[i * 4 + 2] = 0.2;
            px[i * 4 + 3] = 1.0;
        }
        let original = px.clone();
        render(&mut px, n, n, &DEFAULTS, 1.0).unwrap();
        for (a, b) in px.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn glow_only_removes_the_source() {
        let n = 32;
        let mut px = dot_image(n);
        let mut p = DEFAULTS;
        p[11] = 1.0;
        render(&mut px, n, n, &p, 1.0).unwrap();
        assert!(px[3] < 0.5);
    }

    #[test]
    fn source_view_only_applies_opacity() {
        let n = 16;
        let original = dot_image(n);
        let mut px = original.clone();
        let mut p = DEFAULTS;
        p[11] = 2.0;
        p[12] = 50.0;
        render(&mut px, n, n, &p, 1.0).unwrap();
        let expected: Vec<f32> = original.iter().map(|v| v * 0.5).collect();
        assert_eq!(px, expected);
    }

    #[test]
    fn bad_input_is_rejected_without_panicking() {
        let mut px = vec![0.0f32; 16];
        assert!(render(&mut px, 2, 2, &DEFAULTS[..3], 1.0).is_err());
        assert!(render(&mut px, 0, 0, &DEFAULTS, 1.0).is_err());
        let mut tiny = vec![1.0f32; 4];
        assert!(render(&mut tiny, 1, 1, &DEFAULTS, 1.0).is_ok());
    }
}
