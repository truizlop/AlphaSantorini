import { init } from "./wasm/index.js";

window.SantoriniWasmInitPromise = Promise.resolve(init);
