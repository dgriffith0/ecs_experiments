-- Demo Luau script. The host calls `on_update(dt)` every frame and exposes a
-- small `game` API. Here we spawn a ring of characters shortly after start to
-- prove scripts can drive the ECS. Edit freely — it reloads on rebuild.

local elapsed: number = 0.0
local spawned = false

function on_update(dt: number)
    elapsed += dt
    if not spawned and elapsed > 0.5 then
        spawned = true
        game.log("spawning a ring of characters from Luau")
        local count = 8
        for i = 0, count - 1 do
            local a = (i / count) * math.pi * 2.0
            game.spawn("male", 16.0 + math.cos(a) * 5.0, 54.0 + math.sin(a) * 5.0)
        end
    end
end
