import type { InstantiateOptions, ModuleSource } from "../instantiate.js"

export function defaultBrowserSetup(options: {
    module: ModuleSource,
    args?: string[],
    onStdoutLine?: (line: string) => void,
    onStderrLine?: (line: string) => void,
    spawnWorker?: (module: WebAssembly.Module, memory: WebAssembly.Memory, startArg: any) => Worker,
}): Promise<InstantiateOptions>

export function createDefaultWorkerFactory(preludeScript?: string): (module: WebAssembly.Module, memory: WebAssembly.Memory, startArg: any) => Worker
