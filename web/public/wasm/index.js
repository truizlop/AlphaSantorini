// @ts-check
import { instantiate } from './instantiate.js';
import { defaultBrowserSetup , createDefaultWorkerFactory as createDefaultWorkerFactoryForBrowser } from './platforms/browser.js';


/** @type {import('./index.d').init} */
async function initBrowser(_options) {
    /** @type {import('./index.d').Options} */
    const options = _options || {
    };
    let module = options.module;
    if (!module) {
        module = fetch(new URL("SantoriniWasm.wasm", import.meta.url))
    }
    const instantiateOptions = await defaultBrowserSetup({
        module,
        spawnWorker: createDefaultWorkerFactoryForBrowser()
    })
    return await instantiate(instantiateOptions);
}


/** @type {import('./index.d').init} */
export async function init(options) {
        return initBrowser(options);
    }