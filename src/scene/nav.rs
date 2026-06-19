//! Grid navigation mesh derived from the terrain heightmap. Each surface column
//! is a cell; two 4-neighbour cells are linked when the height step between them
//! is small enough to traverse (cliffs break the link). Walkable cells are held
//! in a `pathfinding::grid::Grid` and queried with the crate's A* implementation.

use pathfinding::grid::Grid;
use pathfinding::prelude::astar;

use crate::scene::terrain::Heightmap;

/// Largest height difference (in voxels) an agent can step between adjacent cells.
const NAV_MAX_STEP: i64 = 1;
/// Lift the overlay lines just above the surface so they don't z-fight terrain.
const OVERLAY_LIFT: f32 = 0.05;

/// 4-neighbour offsets (−x, +x, −z, +z).
const NEIGHBOURS: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

/// Walkable surface graph for the current world: a `pathfinding` grid of walkable
/// cells plus the link geometry for the debug overlay.
#[derive(bevy_ecs::prelude::Resource)]
pub struct NavMesh {
    grid: Grid,
    /// Walkable cells (grid coords), for random target selection.
    cells: Vec<(usize, usize)>,
    /// Flat list of line endpoints (pairs) in world space for the overlay.
    segments: Vec<[f32; 3]>,
}

impl NavMesh {
    /// Whether two columns are close enough in height to step between.
    fn linked(hm: &Heightmap, a: (i64, i64), b: (i64, i64)) -> bool {
        (hm.height(a.0, a.1) as i64 - hm.height(b.0, b.1) as i64).abs() <= NAV_MAX_STEP
    }

    pub fn build(hm: &Heightmap) -> Self {
        let extent = hm.extent() as usize;
        let e = extent as i64;
        let mut grid = Grid::new(extent, extent);
        let mut cells = Vec::new();
        let mut segments = Vec::new();

        for gz in 0..e {
            for gx in 0..e {
                let mut any_link = false;
                for &(dx, dz) in &NEIGHBOURS {
                    let (nx, nz) = (gx + dx, gz + dz);
                    if nx < 0 || nz < 0 || nx >= e || nz >= e {
                        continue;
                    }
                    if !Self::linked(hm, (gx, gz), (nx, nz)) {
                        continue;
                    }
                    any_link = true;
                    // Only emit geometry toward +x / +z so each link is drawn once.
                    if dx > 0 || dz > 0 {
                        let mut a = hm.cell_center(gx, gz);
                        let mut b = hm.cell_center(nx, nz);
                        a.y += OVERLAY_LIFT;
                        b.y += OVERLAY_LIFT;
                        segments.push(a.to_array());
                        segments.push(b.to_array());
                    }
                }
                // A cell with no traversable neighbour (isolated pillar/cliff-top)
                // is left out of the graph.
                if any_link {
                    grid.add_vertex((gx as usize, gz as usize));
                    cells.push((gx as usize, gz as usize));
                }
            }
        }

        Self {
            grid,
            cells,
            segments,
        }
    }

    /// Overlay line endpoints (pairs), world space.
    pub fn segments(&self) -> &[[f32; 3]] {
        &self.segments
    }

    /// All walkable cells, for picking random destinations.
    pub fn cells(&self) -> &[(usize, usize)] {
        &self.cells
    }

    /// A* path of cell coords from `start` to `goal` (inclusive of both), or
    /// `None` if unreachable. Adjacent cells only link when the height step is
    /// `≤ NAV_MAX_STEP`, so cliffs aren't crossed. Uniform per-step cost.
    pub fn find_path(
        &self,
        hm: &Heightmap,
        start: (usize, usize),
        goal: (usize, usize),
    ) -> Option<Vec<(usize, usize)>> {
        astar(
            &start,
            |&p| {
                let hp = hm.height(p.0 as i64, p.1 as i64) as i64;
                self.grid
                    .neighbours(p)
                    .into_iter()
                    .filter(move |n| {
                        (hm.height(n.0 as i64, n.1 as i64) as i64 - hp).abs() <= NAV_MAX_STEP
                    })
                    .map(|n| (n, 1usize))
            },
            |&p| self.grid.distance(p, goal),
            |&p| p == goal,
        )
        .map(|(path, _cost)| path)
    }
}
