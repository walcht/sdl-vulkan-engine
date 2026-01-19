# About

## Installation

```bash
cmake -S . -B build/ -G "Ninja" -DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_COMPILER="clang++-21"
cd build/
cmake --build .
```

Optionally add `-DCMAKE_EXPORT_COMPILE_COMMANDS=1 ` to generate the
`compile_commands.json` for your clangd LSP then:

```bash
cp build/compile_commands.json .
```

## Project Structure

```
.
├── build       -> build files
├── doc         -> auto-generated docs
├── external    -> external dependencies (e.g., stb)
├── include     -> public engine headers
├── shaders     -> engine shaders
├── src         -> engine implementation + private headers
│   ├── engine  -> engine core
│   └── gui     -> gui (Qt)
└── test        -> tests
```

## License

MIT License. Read License.txt
