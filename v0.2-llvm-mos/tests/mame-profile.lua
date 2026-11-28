-- MAME autoboot script: boot TINY, ask one question, time it, and sample the
-- PC plus the reconstructed call stack once per periodic callback.
-- Run with a working directory that holds syms.lua, an array of
-- {address, "name"} pairs sorted by address (text symbols of TINY.SYSTEM.elf).
-- Writes stacks.txt, pcs.txt and series.txt there.
-- TINY_QUESTION overrides the question. TINY_CHECK=1 types :check first and
-- prints the position,token,exponent,hash rows instead of timing an answer.
-- TINY_TIMING=1 times inference without sampling or requiring a symbol map.
-- TINY_WM_LO=<hex> measures the soft-stack high-water mark above that address.
local question = os.getenv("TINY_QUESTION") or "WHAT IS THE CAPITAL OF FRANCE?"
local check = os.getenv("TINY_CHECK") == "1"
local profile = not check and os.getenv("TINY_TIMING") ~= "1"
local syms = profile and dofile("syms.lua") or {}
local nsyms = #syms
local wm_lo = tonumber(os.getenv("TINY_WM_LO") or "", 16)

local phase = "load"
local finished = false
local t_ready = nil
local first_visible = false
local snap = nil                          -- code snapshot for JSR validation
local stacks = {}                         -- phase|folded stack -> count
local pcs = {}                            -- infer-phase PC -> count
local series = {}                         -- minute -> {total, attention, matvec}
local samples = 0
local last_poll = -1
local rows, nrows = {}, 0                 -- check mode: position -> row text
local wm_filled = false

local function fname(addr)
    if addr >= 0xC000 then return "<rom>" end
    if addr < syms[1][1] then return "<low>" end
    local lo, hi = 1, nsyms
    while lo < hi do
        local mid = (lo + hi + 1) // 2
        if syms[mid][1] <= addr then lo = mid else hi = mid - 1 end
    end
    return syms[lo][2]
end

local function text_page(space)
    local out = {}
    for row = 0, 23 do
        local base = 0x400 + (row % 8) * 0x80 + (row // 8) * 0x28
        local cols = {}
        for col = 0, 39 do
            local b = space:read_u8(base + col) & 0x7f
            if b < 0x20 then b = b + 0x40 end
            if b == 0x7f then b = 0x20 end
            cols[#cols + 1] = string.char(b)
        end
        out[#out + 1] = (table.concat(cols):gsub("%s+$", ""))
    end
    return table.concat(out, "\n"):gsub("\n+$", "")
end

local function sample(cpu, space, t)
    local pc = cpu.state["PC"].value
    local names = {}
    local in_att, in_mat = false, false
    if snap then
        -- A return address is accepted when a JSR opcode sits two bytes
        -- before it. Pushed data can occasionally be misread as a frame.
        local sp = cpu.state["SP"].value & 0xff
        local frames = {}
        local k = 0x101 + sp
        while k < 0x1ff do
            local ret = space:read_u8(k) | (space:read_u8(k + 1) << 8)
            local site = ret - 2
            if site >= 0x2000 and site <= 0xBFFD and snap[site] == 0x20 then
                frames[#frames + 1] = site
                k = k + 2
            else
                k = k + 1
            end
        end
        local prev = nil
        for i = #frames, 1, -1 do
            local n = fname(frames[i])
            if n ~= prev then names[#names + 1] = n; prev = n end
            if n == "attention" then in_att = true elseif n == "matvec" then in_mat = true end
        end
        local leaf = fname(pc)
        if leaf ~= prev then names[#names + 1] = leaf end
    else
        names[1] = fname(pc)
    end
    local key = phase .. "|" .. table.concat(names, ";")
    stacks[key] = (stacks[key] or 0) + 1
    if phase ~= "load" then
        pcs[pc] = (pcs[pc] or 0) + 1
        samples = samples + 1
        local b = (t - t_ready) // 60
        local s = series[b]
        if not s then s = {0, 0, 0}; series[b] = s end
        s[1] = s[1] + 1
        if in_att then s[2] = s[2] + 1 end
        if in_mat then s[3] = s[3] + 1 end
    end
end

local function dump()
    local f = io.open("stacks.txt", "w")
    for k, n in pairs(stacks) do f:write(n, " ", k, "\n") end
    f:close()
    f = io.open("pcs.txt", "w")
    for pc, n in pairs(pcs) do f:write(("%04X %d\n"):format(pc, n)) end
    f:close()
    f = io.open("series.txt", "w")
    local keys = {}
    for b in pairs(series) do keys[#keys + 1] = b end
    table.sort(keys)
    for _, b in ipairs(keys) do
        local s = series[b]
        f:write(("%d %d %d %d\n"):format(b, s[1], s[2], s[3]))
    end
    f:close()
end

local function finish(m, space, t, screen)
    finished = true
    print(("DONE at %.1f s; inference %.1f s; %d samples"):format(t, t - t_ready, samples))
    if check then
        for p = 0, #question do print(rows[p] or ("MISSING ROW " .. p)) end
    else
        print(screen)
    end
    if wm_lo then
        local a = wm_lo
        while a < 0xBF00 and space:read_u8(a) == 0xA5 do a = a + 1 end
        print(("lowest touched address %04X: soft stack used at most %d bytes"):format(a, 0xBF00 - a))
    end
    if profile then dump() end
    m:exit()
end

emu.register_periodic(function()
    if finished then return end
    local m = manager.machine
    local cpu = m.devices[":maincpu"]
    local space = cpu.spaces["program"]
    local t = m.time:as_double()
    if profile then sample(cpu, space, t) end
    if t < 1 or t < last_poll + 0.5 then return end
    last_poll = t
    local screen = text_page(space)
    if wm_lo and not wm_filled and screen:find("TINY v0.2", 1, true) then
        -- Fill the idle gap below the live soft stack before the model loads.
        wm_filled = true
        local sp = space:read_u8(0) | (space:read_u8(1) << 8)
        for a = wm_lo, sp - 64 do space:write_u8(a, 0xA5) end
        print(("watermark filled %04X..%04X (soft SP %04X)"):format(wm_lo, sp - 64, sp))
    end
    if phase == "load" then
        if screen:find("\n>", 1, true) and screen:find("toggles hash trace", 1, true) then
            if profile then
                snap = {}
                for a = 0x2000, 0xBFFF do snap[a] = space:read_u8(a) end
            end
            t_ready = t
            print(("READY at %.1f s"):format(t))
            if check then
                phase = "arm"
                m.natkeyboard:post(":check\n")
            else
                phase = "infer"
                m.natkeyboard:post(question .. "\n")
            end
        end
    elseif phase == "arm" then
        if screen:find("Trace on", 1, true) then
            phase = "infer"
            t_ready = t
            m.natkeyboard:post(question .. "\n")
        end
    elseif check then
        for line in screen:gmatch("[^\n]+") do
            local p = line:match("^(%d+),%d+,%-?%d+,%x%x%x%x%x%x%x%x$")
            if p and not rows[tonumber(p)] then rows[tonumber(p)] = line; nrows = nrows + 1 end
        end
        if rows[#question] or screen:find("TINY: ", 1, true) then finish(m, space, t, screen) end
    else
        -- The progress dots wrap at column 40 on long prompts.
        local dots, answer = screen:match("Thinking%.%.%.\n([%.\n]+)\n(%S)")
        if not first_visible and dots and #dots:gsub("\n", "") == #question + 1 and answer ~= "[" then
            first_visible = true
            print(("FIRST visible answer at %.1f s; elapsed %.1f s"):format(t, t - t_ready))
        end
        if screen:find(" characters]", 1, true) or screen:find("TINY: ", 1, true) then
            finish(m, space, t, screen)
        end
    end
end)
