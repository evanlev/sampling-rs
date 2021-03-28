use std::sync::mpsc::{channel, RecvError};
use threadpool::ThreadPool;
//use std::sync::Mutex;
use priority_queue::PriorityQueue;
use rand::Rng;
use ordered_float::OrderedFloat;
use clap::Clap;
use rand::thread_rng;
use rand::seq::SliceRandom;

// https://doc.rust-lang.org/book/ch16-01-threads.html
// https://doc.rust-lang.org/book/ch16-03-shared-state.html


/// Generate poisson-disc-like Cartesian sampling patterns. Very basic implementation
/// in progress and intended to be used as a learning exercise, not for clinical or research
/// purposes.
#[derive(Clap)]
#[clap(version = "1.0", author = "Evan Levine <evlevine138e@gmail.com>")]
struct Opts {
    /// Path to write minimum squared distance function.
    #[clap(short, long, required = false, default_value = "")]
    msdf_path: String,
    /// Path to write pattern.
    #[clap(short, long, default_value = "")]
    pat_path: String,
    /// Reduction factor for pattern.
    #[clap(short, long)]
    reduction_factor: f32,
    /// ky encodes.
    #[clap(long)]
    ny: usize,
    /// kz encodes.
    #[clap(long)]
    nz: usize,
    /// blocks in ky to generate in parallel.
    #[clap(long)]
    ky_blocks: usize,
    /// blocks in kz to generate in parallel.
    #[clap(long)]
    kz_blocks: usize,
}

mod io {
    use std::fmt::{Display};
    use std::io::prelude::*;
    use std::fs::File; 
    pub fn write_2d<T: Display>(matrix: &Vec<Vec<T>>, path: &String) -> std::io::Result<()>
    {
        let mut file = File::create(path)?;
        for i in 0..matrix.len()
        {
            for j in 0..matrix[i].len()
            {
                let s = format!("{} ", matrix[i][j]);
                file.write(s.as_ref())?;
            }
            file.write("\n".as_ref())?;
        }
        Ok(())
    }
}

fn is_in_bounds<T>(loc: &(i32, i32), mat: &Vec<Vec<T>>) -> bool
{
    loc.0 >= 0 && loc.1 >= 0 && loc.0 < (mat.len() as i32) && loc.1 < (mat[0].len() as i32)
}

fn walk_horizontal(s: &(i32, i32),
                   loc: & mut (i32, i32),
                   msdf: &mut Vec<Vec<f32>>,
                   msdf_pq: &mut PriorityQueue<(usize, usize), OrderedFloat<f32>>,
                   step_length: u32,
                   large_sqdist_threshold: f32,
                   dir: i32) -> bool
{
    let dz = s.1 - loc.1;
    let dz_sq = (dz * dz) as f32;
    let mut updated: bool = false;
    for _i in 0..step_length
    {
        loc.0 += dir;
        let locu = (loc.0 as usize, loc.1 as usize);
        if is_in_bounds(&loc, &msdf)
        {
            let cur_dsq = msdf[locu.0][locu.1];
            if dz_sq >= cur_dsq
            {
                continue;
            }
            let dy = loc.0 - (s.0 as i32);
            let new_dsq = dz_sq + (dy * dy) as f32;
            if new_dsq < cur_dsq && new_dsq < large_sqdist_threshold
            {
                msdf[locu.0][locu.1] = new_dsq;
                msdf_pq.change_priority(&locu, OrderedFloat(new_dsq));
                updated = true;
            }
        }
    }
    updated
}

fn walk_vertical(s: &(i32, i32),
                 loc: & mut (i32, i32),
                 msdf: &mut Vec<Vec<f32>>,
                 msdf_pq: &mut PriorityQueue<(usize, usize), OrderedFloat<f32>>,
                 step_length: u32,
                 large_sqdist_threshold: f32,
                 dir: i32) -> bool
{
    let dy = s.0 - loc.0;
    let dy_sq = (dy * dy) as f32;
    let mut updated: bool = false;
    for _i in 0..step_length
    {
        loc.1 += dir;
        let locu = (loc.0 as usize, loc.1 as usize);
        if is_in_bounds(&loc, &msdf) && dy_sq < msdf[locu.0][locu.1]
        {
            let cur_dsq = msdf[locu.0][locu.1];
            if dy_sq >= cur_dsq
            {
                continue;
            }
            let dz = loc.1 - (s.1 as i32);
            let new_dsq = dy_sq + (dz * dz) as f32;
            if new_dsq < cur_dsq && new_dsq < large_sqdist_threshold
            {
                msdf[locu.0][locu.1] = new_dsq;
                msdf_pq.change_priority(&locu, OrderedFloat(new_dsq));
                updated = true;
            }
        }
    }
    updated
}

/// Generate a pattern with a given reduction factor.
///
/// # Arguments
///
/// * `reduction_factor` - Reduction factor for the pattern
/// * `pat` - Pattern to fill
/// * `msdf` - "Minimum squared-distance function" containing the
///            squared distance to the nearest sample, 0 at the sample
///            locations.
fn generate_pattern(opts: &Opts, reduction_factor: f32, pat: &mut Vec<Vec<u32>>, msdf: &mut Vec<Vec<f32>>) -> Result<(), RecvError>
{
    let large_sqdist_threshold = 2.0f32 * reduction_factor;

    let n_threads = opts.ky_blocks * opts.kz_blocks;
    let ny = pat.len() / opts.ky_blocks;
    let nz = pat[0].len() / opts.kz_blocks;
    let size = ny * nz;
    let max_samples = ((size as f32) / reduction_factor) as u32;
    
    let pool = ThreadPool::new(n_threads);
    let (tx, rx) = channel();

    for idx_block_y in 0..opts.ky_blocks {
        for idx_block_z in 0..opts.kz_blocks {
            let tx = tx.clone();
            pool.execute(move || {  
                let mut rng = rand::thread_rng();

                // Pattern for this block
                let mut pat_block = vec![vec![0u32; nz]; ny];
                
                // Initialize the MSDF
                let mut sum = 0;
                let max_dist = (ny * ny + nz * nz) as f32;
                let mut msdf_block = vec![vec![max_dist; nz]; ny]; // Min (squared) distance function
                let mut msdf_pq = PriorityQueue::new();
                let mut ny_random_range: Vec<usize> = (0..ny).collect();
                ny_random_range.shuffle(&mut thread_rng());
                for y in ny_random_range
                {
                    let mut nz_random_range: Vec<usize> = (0..nz).collect();
                    nz_random_range.shuffle(&mut thread_rng());
                    for z in nz_random_range
                    {
                        msdf_block[y][z] = max_dist + (rng.gen_range(0.0..1e-2) as f32);
                        msdf_pq.push((y, z), OrderedFloat(msdf_block[y][z]));
                    }
                }
                
                while sum < max_samples
                {
                    let best = msdf_pq.peek();
                    match best
                    {
                        Some((&(sy, sz), &_priority)) => {
                            pat_block[sy][sz] += 1;
                            msdf_block[sy][sz] = 0f32;
                            msdf_pq.change_priority(&(sy,sz), OrderedFloat(0f32));

                            // Update MSDF outward in spiral order
                            let mut step_length = 1;
                            let mut loc = (sy as i32, sz as i32);
                            let ref s = (sy as i32, sz as i32);

                            loop
                            {
                                let updated1 = walk_horizontal(s, &mut loc, &mut msdf_block, &mut msdf_pq, step_length, large_sqdist_threshold, 1);
                                let updated2 = walk_vertical(s, &mut loc, &mut msdf_block, &mut msdf_pq, step_length, large_sqdist_threshold, 1);
                                step_length += 1;
                                let updated3 = walk_horizontal(s, &mut loc, &mut msdf_block, &mut msdf_pq, step_length, large_sqdist_threshold, -1);
                                let updated4 = walk_vertical(s, &mut loc, &mut msdf_block, &mut msdf_pq, step_length, large_sqdist_threshold, -1);
                                step_length += 1;
                                if !(updated1 || updated2 || updated3 || updated4)
                                {
                                    break;
                                }
                            }
                        },
                        None => break,
                    }
                    sum += 1;
                }
                tx.send((idx_block_y, idx_block_z, msdf_block, pat_block)).expect("Could not send data!");
            });
        }
    }

    // Concatenate patterns.
    for _block_ky in 0..opts.ky_blocks {
        for _block_kz in 0..opts.kz_blocks {
            let (idx_block_y, idx_block_z, msdf_block, pat_block) = rx.recv()?;
            let ky_offset = ny * idx_block_y;
            let kz_offset = nz * idx_block_z;

            for ky in 0..ny
            {
                for kz in 0..nz
                {
                    pat[ky + ky_offset][kz+kz_offset] = pat_block[ky][kz];
                    msdf[ky + ky_offset][kz+kz_offset] = msdf_block[ky][kz];
                }
            }
        }
    }

    Ok(())
}


fn main() -> std::io::Result<()> {
    let opts: Opts = Opts::parse();
    println!("Pattern size: {ny} x {nz}, Reduction factor: {r}", ny=opts.ny, nz=opts.nz, r=opts.reduction_factor);

    let mut pat = vec![vec![0u32; opts.nz]; opts.ny];
    let mut msdf = vec![vec![0f32; opts.nz]; opts.ny];

    let res = generate_pattern(&opts, opts.reduction_factor, &mut pat, &mut msdf);
    match res
    {
        Err(_e) => panic!("Failed to generate pattern!"),
        Ok(_n) => println!("Pattern generated!"),
    };

    if opts.pat_path != ""
    {
        io::write_2d(&pat, &opts.pat_path)?;
    }
    if opts.msdf_path != ""
    {
        io::write_2d(&msdf, &opts.msdf_path)?
    }
 
    Ok(())
}
