# About

## Installation

```bash
cmake -S . -B bin/ -G "Ninja" -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_COMPILER="clang++"
cd build/ && ./main
```

Optionally add `-DCMAKE_EXPORT_COMPILE_COMMANDS=1 ` to generate the
`compile_commands.json` for your clangd LSP.
