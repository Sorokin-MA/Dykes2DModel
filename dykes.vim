let SessionLoad = 1
let s:so_save = &g:so | let s:siso_save = &g:siso | setg so=0 siso=0 | setl so=-1 siso=-1
let v:this_session=expand("<sfile>:p")
silent only
silent tabonly
cd c:/WORKSPACE/dykes_julia/Dykes2DModel
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
let s:shortmess_save = &shortmess
if &shortmess =~ 'A'
  set shortmess=aoOA
else
  set shortmess=aoO
endif
badd +1 test.jl
badd +0 term://c:/WORKSPACE/dykes_julia/Dykes2DModel//19940:C:/Windows/system32/cmd.exe
badd +0 term://c:/WORKSPACE/dykes_julia/Dykes2DModel//7564:C:/Windows/system32/cmd.exe
argglobal
%argdel
$argadd test.jl
edit test.jl
let s:save_splitbelow = &splitbelow
let s:save_splitright = &splitright
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
3wincmd h
wincmd w
wincmd _ | wincmd |
split
1wincmd k
wincmd w
wincmd w
wincmd w
let &splitbelow = s:save_splitbelow
let &splitright = s:save_splitright
wincmd t
let s:save_winminheight = &winminheight
let s:save_winminwidth = &winminwidth
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 31 + 115) / 230)
exe '2resize ' . ((&lines * 30 + 31) / 62)
exe 'vert 2resize ' . ((&columns * 78 + 115) / 230)
exe '3resize ' . ((&lines * 29 + 31) / 62)
exe 'vert 3resize ' . ((&columns * 78 + 115) / 230)
exe 'vert 4resize ' . ((&columns * 78 + 115) / 230)
exe 'vert 5resize ' . ((&columns * 40 + 115) / 230)
argglobal
enew
file NERD_tree_1
balt test.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
argglobal
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
silent! normal! zE
let &fdl = &fdl
let s:l = 12 - ((11 * winheight(0) + 15) / 30)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 12
normal! 03|
wincmd w
argglobal
if bufexists(fnamemodify("term://c:/WORKSPACE/dykes_julia/Dykes2DModel//7564:C:/Windows/system32/cmd.exe", ":p")) | buffer term://c:/WORKSPACE/dykes_julia/Dykes2DModel//7564:C:/Windows/system32/cmd.exe | else | edit term://c:/WORKSPACE/dykes_julia/Dykes2DModel//7564:C:/Windows/system32/cmd.exe | endif
if &buftype ==# 'terminal'
  silent file term://c:/WORKSPACE/dykes_julia/Dykes2DModel//7564:C:/Windows/system32/cmd.exe
endif
balt test.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 4 - ((3 * winheight(0) + 14) / 29)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 4
normal! 0
wincmd w
argglobal
if bufexists(fnamemodify("term://c:/WORKSPACE/dykes_julia/Dykes2DModel//19940:C:/Windows/system32/cmd.exe", ":p")) | buffer term://c:/WORKSPACE/dykes_julia/Dykes2DModel//19940:C:/Windows/system32/cmd.exe | else | edit term://c:/WORKSPACE/dykes_julia/Dykes2DModel//19940:C:/Windows/system32/cmd.exe | endif
if &buftype ==# 'terminal'
  silent file term://c:/WORKSPACE/dykes_julia/Dykes2DModel//19940:C:/Windows/system32/cmd.exe
endif
balt test.jl
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 29 - ((28 * winheight(0) + 30) / 60)
if s:l < 1 | let s:l = 1 | endif
keepjumps exe s:l
normal! zt
keepjumps 29
normal! 0
wincmd w
argglobal
enew
file __Tagbar__.1
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal nofen
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 31 + 115) / 230)
exe '2resize ' . ((&lines * 30 + 31) / 62)
exe 'vert 2resize ' . ((&columns * 78 + 115) / 230)
exe '3resize ' . ((&lines * 29 + 31) / 62)
exe 'vert 3resize ' . ((&columns * 78 + 115) / 230)
exe 'vert 4resize ' . ((&columns * 78 + 115) / 230)
exe 'vert 5resize ' . ((&columns * 40 + 115) / 230)
tabnext 1
if exists('s:wipebuf') && len(win_findbuf(s:wipebuf)) == 0 && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20
let &shortmess = s:shortmess_save
let &winminheight = s:save_winminheight
let &winminwidth = s:save_winminwidth
let s:sx = expand("<sfile>:p:r")."x.vim"
if filereadable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &g:so = s:so_save | let &g:siso = s:siso_save
set hlsearch
nohlsearch
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
