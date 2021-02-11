set nocompatible              " required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'
" Plugin 'vim-syntastic/syntastic'
Plugin 'ycm-core/YouCompleteMe'
" Plugin 'artur-shaik/vim-javacomplete2'
" Plugin 'nvie/vim-flake8'
" Plugin 'davidhalter/jedi-vim'
"Plugin 'vim-airline/vim-airline'

" add all your plugins here (note older versions of Vundle
" used Bundle instead of Plugin)

" ...

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required


" set UTF-8 encoding
set enc=utf-8
set fenc=utf-8
set termencoding=utf-8
" disable vi compatibility (emulation of old bugs)
set nocompatible
" use indentation of previous line
set autoindent
" use intelligent indentation for C
set smartindent
" configure tabwidth and insert spaces instead of tabs
set tabstop=4        " tab width is 4 spaces
set shiftwidth=4     " indent also with 4 spaces
set softtabstop=4
set expandtab        " expand tabs to spaces
" wrap lines at 120 chars. 80 is somewaht antiquated with nowadays displays.
set textwidth=120
" turn syntax highlighting on
set t_Co=256
syntax on
" turn line numbers on
" set number                "absolute
set number relativenumber   " hybrid
" highlight matching braces
set showmatch
" set increamental search
set is

function! ConditionalPairMap(open, close)
  let line = getline('.')
  let col = col('.')
  if col < col('$') || stridx(line, a:close, col + 1) != -1
    return a:open
  else
    return a:open . a:close . repeat("\<left>", len(a:close))
  endif
endf
inoremap <expr> ( ConditionalPairMap('(', ')')
inoremap <expr> )  strpart(getline('.'), col('.')-1, 1) == ")" ? "\<Right>" : ")"
inoremap <expr> [ ConditionalPairMap('[', ']')
inoremap <expr> ]  strpart(getline('.'), col('.')-1, 1) == "]" ? "\<Right>" : "]"
inoremap <expr> { ConditionalPairMap('{', '}')
inoremap {<CR>  {<CR>}<Esc>O
inoremap {}     {}

inoremap <c-a> <Esc>ggVG<CR>

vnoremap ( <Esc>`>a)<Esc>`<i(<Esc>
vnoremap ' <Esc>`>a'<Esc>`<i'<Esc>
vnoremap " <Esc>`>a"<Esc>`<i"<Esc>
vnoremap [ <Esc>`>a]<Esc>`<i[<Esc>
"copy pasting with indent on
" noremap p ]p

" Syntastic settings
let g:ycm_global_ycm_extra_conf = "~/.vim/bundle/YouCompleteMe/.ycm_extra_conf.py"

autocmd FileType java setlocal omnifunc=javacomplete#Complete
nmap <F4> <Plug>(JavaComplete-Imports-AddSmart)

imap <F4> <Plug>(JavaComplete-Imports-AddSmart)

"au BufRead,BufNewFile *.py,*.pyw,*.c,*.h match BadWhitespace /\s\+$/

set completeopt-=preview

:autocmd BufNewFile *.cpp 0r ~/.vim/templates/skeleton.cpp

set clipboard=unnamedplus

"For commenting
let s:comment_map = { 
    \   "c": '\/\/',
    \   "cpp": '\/\/',
    \   "go": '\/\/',
    \   "java": '\/\/',
    \   "javascript": '\/\/',
    \   "lua": '--',
    \   "scala": '\/\/',
    \   "php": '\/\/',
    \   "python": '#',
    \   "ruby": '#',
    \   "rust": '\/\/',
    \   "sh": '#',
    \   "desktop": '#',
    \   "fstab": '#',
    \   "conf": '#',
    \   "profile": '#',
    \   "bashrc": '#',
    \   "bash_profile": '#',
    \   "mail": '>',
    \   "eml": '>',
    \   "bat": 'REM',
    \   "ahk": ';',
    \   "vim": '"',
    \   "tex": '%',
    \ }

function! ToggleComment()
    if has_key(s:comment_map, &filetype)
        let comment_leader = s:comment_map[&filetype]
        if getline('.') =~ "^\\s*" . comment_leader . " " 
            " Uncomment the line
            execute "silent s/^\\(\\s*\\)" . comment_leader . " /\\1/"
        else 
            if getline('.') =~ "^\\s*" . comment_leader
                " Uncomment the line
                execute "silent s/^\\(\\s*\\)" . comment_leader . "/\\1/"
            else
                " Comment the line
                execute "silent s/^\\(\\s*\\)/\\1" . comment_leader . " /"
            end
        end
    else
        echo "No comment leader found for filetype"
    end
endfunction


nnoremap cc :call ToggleComment()<cr>
vnoremap cc :call ToggleComment()<cr>

" StatusLine
set laststatus=2
set statusline=%1*
set statusline+=\ \ %m\ \ 
set statusline+=%*
set statusline+=\ %<%F\ 
set statusline+=\ %y
set statusline+=%=
set statusline+=%#CursorColumn#
" set statusline+=\[%{&fileformat}\]
set statusline+=%4*
set statusline+=\ %{&fileencoding?&fileencoding:&encoding}
set statusline+=\ line:%l/%L,\ col:%-2c
set statusline+=\ %p%%
set statusline+=\ \ \ 
set statusline+=%1*

set background=dark
if &background ==# 'dark'
    colorscheme monokai
else
    colorscheme default
end
