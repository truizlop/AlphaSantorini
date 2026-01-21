import * as THREE from "three";
import type { StateSummary } from "../engine/wasmBridge";
import { BOARD_SIZE, decodeAction, destination, tileKey } from "../game/actions";

const TILE_SIZE = 1;
const TILE_BASE_HEIGHT = 0.04;
const TILE_TOP_HEIGHT = 0.06;
const TILE_TOP = TILE_BASE_HEIGHT + TILE_TOP_HEIGHT;
const LEVEL_HEIGHT = 0.22;
const LEVEL_SIZES = [0.9, 0.76, 0.62];
const LEVEL_COLORS = [0xf7f5ef, 0xf1eee6, 0xeae4d8];

export type HighlightType = "placement" | "move" | "build";

export class BoardScene {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private raycaster: THREE.Raycaster;
  private pointer: THREE.Vector2;
  private boardGroup: THREE.Group;
  private buildingsGroup: THREE.Group;
  private workersGroup: THREE.Group;
  private effectsGroup: THREE.Group;
  private tileMeshes: THREE.Mesh[] = [];
  private highlightMap: Map<string, HighlightType> = new Map();
  private hoverKey: string | null = null;
  private selectedWorkerKey: string | null = null;
  private onTileClick: (row: number, col: number) => void;
  private startTime = performance.now();
  private dragPointerId: number | null = null;
  private dragStart = new THREE.Vector2();
  private dragLast = new THREE.Vector2();
  private dragMoved = false;
  private rotationTarget = 0;

  constructor(container: HTMLElement, onTileClick: (row: number, col: number) => void) {
    this.container = container;
    this.onTileClick = onTileClick;
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xd7eef4, 5, 18);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
    this.camera.position.set(4.2, 5.2, 6.6);
    this.camera.lookAt(0, 0, 0);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();

    this.boardGroup = new THREE.Group();
    this.buildingsGroup = new THREE.Group();
    this.workersGroup = new THREE.Group();
    this.effectsGroup = new THREE.Group();
    this.boardGroup.add(this.buildingsGroup);
    this.boardGroup.add(this.workersGroup);
    this.boardGroup.add(this.effectsGroup);
    this.scene.add(this.boardGroup);

    this.addLights();
    this.buildBoard();
    this.bindEvents();
    this.handleResize();
    this.animate();
  }

  update(summary: StateSummary, highlightMap: Map<string, HighlightType>, selectedWorkerKey: string | null): void {
    this.highlightMap = highlightMap;
    this.selectedWorkerKey = selectedWorkerKey;
    this.updateBuildings(summary);
    this.updateWorkers(summary);
    this.clearGroup(this.effectsGroup);
    this.applyHighlights();
  }

  private buildBoard(): void {
    const tileTopGeo = new THREE.BoxGeometry(0.9, TILE_TOP_HEIGHT, 0.9);
    const tileBaseGeo = new THREE.BoxGeometry(0.98, TILE_BASE_HEIGHT, 0.98);
    const topColor = new THREE.Color(0xf8f4ea);
    const tileBaseMat = new THREE.MeshStandardMaterial({
      color: 0xe7dcc8,
      roughness: 0.9,
      metalness: 0.05,
    });

    for (let row = 0; row < BOARD_SIZE; row += 1) {
      for (let col = 0; col < BOARD_SIZE; col += 1) {
        const jitter = ((row * 7 + col * 11) % 5 - 2) * 0.008;
        const mat = new THREE.MeshStandardMaterial({
          color: topColor.clone().offsetHSL(0, -0.04, jitter),
          roughness: 0.85,
          metalness: 0.05,
        });
        const tileGroup = new THREE.Group();
        const tileBase = new THREE.Mesh(tileBaseGeo, tileBaseMat);
        tileBase.position.y = TILE_BASE_HEIGHT / 2;
        tileBase.castShadow = false;
        tileBase.receiveShadow = true;
        const tileTop = new THREE.Mesh(tileTopGeo, mat);
        tileTop.position.y = TILE_BASE_HEIGHT + TILE_TOP_HEIGHT / 2;
        tileTop.castShadow = false;
        tileTop.receiveShadow = true;
        tileTop.userData = { row, col, baseColor: mat.color.clone() };
        tileGroup.add(tileBase);
        tileGroup.add(tileTop);
        tileGroup.position.set((col - 2) * TILE_SIZE, 0, (row - 2) * TILE_SIZE);
        this.tileMeshes.push(tileTop);
        this.boardGroup.add(tileGroup);
      }
    }

    const waterGeo = new THREE.BoxGeometry(5.2, 0.04, 5.2);
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x8ac6d7,
      roughness: 0.6,
      metalness: 0.1,
    });
    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.y = -0.025;
    water.receiveShadow = true;
    this.boardGroup.add(water);

    const frameGeo = new THREE.BoxGeometry(5.9, 0.12, 5.9);
    const frameMat = new THREE.MeshStandardMaterial({
      color: 0xf0d9b8,
      roughness: 0.9,
      metalness: 0.05,
    });
    const frame = new THREE.Mesh(frameGeo, frameMat);
    frame.position.y = -0.1;
    frame.receiveShadow = true;
    this.boardGroup.add(frame);

    const baseGeo = new THREE.BoxGeometry(6.4, 0.6, 6.4);
    const boardBaseMat = new THREE.MeshStandardMaterial({
      color: 0x6bb2c2,
      roughness: 0.6,
      metalness: 0.1,
    });
    const base = new THREE.Mesh(baseGeo, boardBaseMat);
    base.position.y = -0.46;
    base.receiveShadow = true;
    this.boardGroup.add(base);
  }

  private updateBuildings(summary: StateSummary): void {
    this.clearGroup(this.buildingsGroup);

    summary.boardHeights.forEach((height, index) => {
      const row = Math.floor(index / BOARD_SIZE);
      const col = index % BOARD_SIZE;
      const levels = Math.min(height, 3);
      const x = (col - 2) * TILE_SIZE;
      const z = (row - 2) * TILE_SIZE;
      for (let level = 0; level < levels; level += 1) {
        const stack = this.createBuildingLevel(level);
        stack.position.set(x, TILE_TOP + LEVEL_HEIGHT / 2 + level * LEVEL_HEIGHT, z);
        this.buildingsGroup.add(stack);
      }
      if (height >= 4) {
        const dome = this.createDome();
        dome.position.set(x, TILE_TOP + LEVEL_HEIGHT * 3, z);
        this.buildingsGroup.add(dome);
      }
    });
  }

  private updateWorkers(summary: StateSummary): void {
    this.clearGroup(this.workersGroup);
    summary.workers.forEach((worker) => {
      const heightIndex = summary.boardHeights[worker.row * BOARD_SIZE + worker.col] ?? 0;
      const isSelected = this.selectedWorkerKey === `${worker.player}-${worker.id}`;
      const group = this.createWorkerMesh(worker, heightIndex, isSelected);
      if (isSelected) {
        group.scale.setScalar(1.18);
      }
      this.workersGroup.add(group);
    });
  }

  private applyHighlights(): void {
    const defaultBase = new THREE.Color(0xf8f4ea);
    const placement = new THREE.Color(0x9ad8b0);
    const move = new THREE.Color(0x8dd1dc);
    const build = new THREE.Color(0xf6b26b);
    const hover = new THREE.Color(0xfff1c2);

    for (const tile of this.tileMeshes) {
      const mat = tile.material as THREE.MeshStandardMaterial;
      const key = tileKey(tile.userData.row, tile.userData.col);
      const baseColor = tile.userData.baseColor as THREE.Color | undefined;
      let color = baseColor ? baseColor.clone() : defaultBase.clone();
      const highlight = this.highlightMap.get(key);
      if (highlight === "placement") color = placement.clone();
      if (highlight === "move") color = move.clone();
      if (highlight === "build") color = build.clone();
      if (this.hoverKey === key) {
        color.lerp(hover, 0.6);
      }
      mat.color = color;
      mat.needsUpdate = true;
    }
  }

  private bindEvents(): void {
    this.container.appendChild(this.renderer.domElement);
    this.container.addEventListener("pointerdown", (event) => this.handlePointerDown(event));
    this.container.addEventListener("pointermove", (event) => this.handlePointerMove(event));
    this.container.addEventListener("pointerup", (event) => this.handlePointerUp(event));
    this.container.addEventListener("pointercancel", (event) => this.handlePointerUp(event));
    window.addEventListener("resize", () => this.handleResize());
  }

  private handlePointer(event: PointerEvent | MouseEvent): void {
    this.updatePointerFromEvent(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const workerHit = this.pickWorkerHit();
    if (workerHit) {
      this.hoverKey = tileKey(workerHit.row, workerHit.col);
    } else {
      const tileHit = this.pickTileHit();
      this.hoverKey = tileHit ? tileKey(tileHit.row, tileHit.col) : null;
    }
    this.applyHighlights();
  }

  private handleClick(event: PointerEvent | MouseEvent): void {
    this.updatePointerFromEvent(event);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const workerHit = this.pickWorkerHit();
    if (workerHit) {
      this.onTileClick(workerHit.row, workerHit.col);
      return;
    }
    const tileHit = this.pickTileHit();
    if (tileHit) {
      this.onTileClick(tileHit.row, tileHit.col);
    } else {
      this.hoverKey = null;
    }
  }

  private handlePointerDown(event: PointerEvent): void {
    if (this.dragPointerId !== null) {
      return;
    }
    this.dragPointerId = event.pointerId;
    this.dragMoved = false;
    this.dragStart.set(event.clientX, event.clientY);
    this.dragLast.copy(this.dragStart);
    this.container.setPointerCapture(event.pointerId);
  }

  private handlePointerMove(event: PointerEvent): void {
    if (this.dragPointerId === event.pointerId) {
      const dx = event.clientX - this.dragLast.x;
      const dy = event.clientY - this.dragLast.y;
      this.dragLast.set(event.clientX, event.clientY);

      if (!this.dragMoved) {
        const moved = Math.hypot(
          event.clientX - this.dragStart.x,
          event.clientY - this.dragStart.y
        );
        if (moved > 4) {
          this.dragMoved = true;
        }
      }

      if (this.dragMoved) {
        const rotationSpeed = 0.003;
        this.rotationTarget += dx * rotationSpeed;
        // Vertical drag could be used for tilt later; keep for future.
        void dy;
      } else {
        this.handlePointer(event);
      }
      return;
    }

    this.handlePointer(event);
  }

  private handlePointerUp(event: PointerEvent): void {
    if (this.dragPointerId !== event.pointerId) {
      return;
    }
    this.container.releasePointerCapture(event.pointerId);
    this.dragPointerId = null;
    if (!this.dragMoved) {
      this.handleClick(event);
    }
    this.dragMoved = false;
  }

  private handleResize(): void {
    const { clientWidth, clientHeight } = this.container;
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(clientWidth, clientHeight);
  }

  private addLights(): void {
    const hemi = new THREE.HemisphereLight(0xfdf7ed, 0x7fb8c9, 0.8);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 0.95);
    dir.position.set(6, 8, 4);
    dir.castShadow = true;
    dir.shadow.mapSize.set(2048, 2048);
    dir.shadow.bias = -0.00025;
    dir.shadow.camera.left = -6;
    dir.shadow.camera.right = 6;
    dir.shadow.camera.top = 6;
    dir.shadow.camera.bottom = -6;
    dir.shadow.camera.near = 1;
    dir.shadow.camera.far = 20;
    this.scene.add(dir);

    const fill = new THREE.DirectionalLight(0xbfdbe6, 0.35);
    fill.position.set(-6, 4, -5);
    this.scene.add(fill);
  }

  private animate(): void {
    requestAnimationFrame(() => this.animate());
    const elapsed = (performance.now() - this.startTime) / 1000;
    const lift = Math.min(elapsed / 1.2, 1);
    this.boardGroup.position.y = -0.6 + lift * 0.6;
    const rotationDelta = this.rotationTarget - this.boardGroup.rotation.y;
    this.boardGroup.rotation.y += rotationDelta * 0.08;
    this.renderer.render(this.scene, this.camera);
  }

  private updatePointerFromEvent(event: PointerEvent | MouseEvent): void {
    const rect = this.container.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  private pickWorkerHit(): { row: number; col: number } | null {
    const hits = this.raycaster.intersectObjects(this.workersGroup.children, true);
    for (const hit of hits) {
      let object: THREE.Object3D | null = hit.object;
      while (object) {
        const row = object.userData?.row;
        const col = object.userData?.col;
        if (Number.isInteger(row) && Number.isInteger(col)) {
          return { row, col };
        }
        object = object.parent;
      }
    }
    return null;
  }

  private pickTileHit(): { row: number; col: number } | null {
    const hits = this.raycaster.intersectObjects(this.tileMeshes, false);
    if (hits.length === 0) {
      return null;
    }
    const { row, col } = hits[0].object.userData as { row: number; col: number };
    return { row, col };
  }

  async animateAction(prev: StateSummary, actionId: number, next: StateSummary): Promise<void> {
    const action = decodeAction(actionId);
    const emptyHighlights = new Map<string, HighlightType>();
    this.update(prev, emptyHighlights, null);

    if (action.type === "placement") {
      const prevKeys = new Set(prev.workers.map((worker) => this.workerKey(worker)));
      const added = next.workers.find((worker) => !prevKeys.has(this.workerKey(worker)));
      if (!added) {
        return;
      }
      const heightIndex = next.boardHeights[added.row * BOARD_SIZE + added.col] ?? 0;
      const group = this.createWorkerMesh(added, heightIndex);
      group.scale.setScalar(0.01);
      this.workersGroup.add(group);
      const startY = group.position.y;
      await this.animateFor(220, (t) => {
        const eased = this.easeOutCubic(t);
        const scale = 0.2 + 0.8 * eased;
        group.scale.setScalar(scale);
        group.position.y = startY + Math.sin(Math.PI * t) * 0.06;
      });
      return;
    }

    const mover = prev.workers.find(
      (worker) => worker.player === prev.turn && worker.id === action.workerId
    );
    if (!mover) {
      return;
    }
    const moveDest = destination(mover.row, mover.col, action.moveDir);
    const buildDest = destination(moveDest.row, moveDest.col, action.buildDir);

    const workerKey = this.workerKey(mover);
    let workerGroup = this.findWorkerGroup(workerKey);
    if (!workerGroup) {
      this.update(prev, emptyHighlights, null);
      workerGroup = this.findWorkerGroup(workerKey);
    }
    if (!workerGroup) {
      return;
    }

    const startPos = workerGroup.position.clone();
    const moveHeight = prev.boardHeights[moveDest.row * BOARD_SIZE + moveDest.col] ?? 0;
    const endPos = this.tileToWorld(moveDest.row, moveDest.col, moveHeight);
    const baseY = endPos.y;

    await this.animateFor(300, (t) => {
      const eased = this.easeOutCubic(t);
      workerGroup!.position.lerpVectors(startPos, endPos, eased);
      workerGroup!.position.y = baseY + Math.sin(Math.PI * t) * 0.12;
    });

    const buildIndex = buildDest.row * BOARD_SIZE + buildDest.col;
    const newHeight = next.boardHeights[buildIndex] ?? 0;
    if (newHeight <= 0) {
      return;
    }
    const buildMesh = this.createBuildMesh(buildDest.row, buildDest.col, newHeight);
    if (!buildMesh) {
      return;
    }
    this.effectsGroup.add(buildMesh);
    await this.animateFor(220, (t) => {
      const eased = this.easeOutCubic(t);
      const scale = Math.max(0.05, eased);
      if (buildMesh.userData.scaleMode === "uniform") {
        buildMesh.scale.setScalar(scale);
      } else {
        buildMesh.scale.y = scale;
      }
      buildMesh.position.y = buildMesh.userData.baseY - (1 - eased) * 0.18;
    });
    this.clearGroup(this.effectsGroup);
  }

  private createWorkerMesh(
    worker: StateSummary["workers"][number],
    heightIndex: number,
    isSelected = false
  ): THREE.Group {
    const tileSize = TILE_SIZE;
    const levelHeight = LEVEL_HEIGHT;
    const tileTop = TILE_TOP;
    const bodyGeo = new THREE.CylinderGeometry(0.18, 0.22, 0.38, 16);
    const headGeo = new THREE.SphereGeometry(0.16, 16, 12);

    const color = worker.player === "one" ? 0x2c8f8f : 0xe07a3a;
    const accent = worker.id === "one" ? 0xffffff : 0x2a2c2f;

    const bodyMat = new THREE.MeshStandardMaterial({ color });
    const headMat = new THREE.MeshStandardMaterial({ color: accent });

    const group = new THREE.Group();
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.castShadow = true;
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 0.3;
    head.castShadow = true;

    group.add(body);
    group.add(head);
    if (isSelected) {
      this.addSelectionRing(group, worker.player);
    }
    const standHeight = tileTop + Math.min(heightIndex, 4) * levelHeight + 0.2;
    group.position.set((worker.col - 2) * tileSize, standHeight, (worker.row - 2) * tileSize);
    group.userData = { key: this.workerKey(worker), row: worker.row, col: worker.col };
    return group;
  }

  private addSelectionRing(group: THREE.Group, player: "one" | "two"): void {
    const ringGeo = new THREE.TorusGeometry(0.33, 0.045, 12, 24);
    const ringColor = player === "one" ? 0x5ed7cb : 0xf2a564;
    const ringMat = new THREE.MeshStandardMaterial({
      color: ringColor,
      emissive: ringColor,
      emissiveIntensity: 0.6,
      transparent: true,
      opacity: 0.85,
      roughness: 0.4,
      metalness: 0.1,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -0.2;
    ring.receiveShadow = true;
    group.add(ring);
  }

  private createBuildingLevel(level: number): THREE.Group {
    const size = LEVEL_SIZES[Math.min(level, LEVEL_SIZES.length - 1)];
    const ledgeSize = size + 0.08;
    const bodyGeo = new THREE.BoxGeometry(size, LEVEL_HEIGHT, size);
    const ledgeGeo = new THREE.BoxGeometry(ledgeSize, 0.04, ledgeSize);
    const bodyMat = new THREE.MeshStandardMaterial({
      color: LEVEL_COLORS[Math.min(level, LEVEL_COLORS.length - 1)],
      roughness: 0.82,
      metalness: 0.04,
    });
    const ledgeMat = new THREE.MeshStandardMaterial({
      color: 0xe4d8c6,
      roughness: 0.88,
      metalness: 0.05,
    });

    const group = new THREE.Group();
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.castShadow = true;
    body.receiveShadow = true;
    const ledge = new THREE.Mesh(ledgeGeo, ledgeMat);
    ledge.position.y = LEVEL_HEIGHT / 2 - 0.02;
    ledge.castShadow = true;
    ledge.receiveShadow = true;
    group.add(body);
    group.add(ledge);
    return group;
  }

  private createDome(): THREE.Group {
    const domeRadius = 0.34;
    const baseHeight = 0.06;
    const domeGeo = new THREE.SphereGeometry(
      domeRadius,
      24,
      16,
      0,
      Math.PI * 2,
      0,
      Math.PI / 2
    );
    const baseGeo = new THREE.CylinderGeometry(domeRadius * 0.92, domeRadius, baseHeight, 24);
    const domeMat = new THREE.MeshPhysicalMaterial({
      color: 0x6fbad0,
      roughness: 0.25,
      metalness: 0.05,
      clearcoat: 0.6,
      clearcoatRoughness: 0.35,
    });
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x5faec4,
      roughness: 0.4,
      metalness: 0.1,
    });

    const group = new THREE.Group();
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.position.y = baseHeight / 2;
    base.castShadow = true;
    base.receiveShadow = true;
    const dome = new THREE.Mesh(domeGeo, domeMat);
    dome.position.y = baseHeight;
    dome.castShadow = true;
    group.add(base);
    group.add(dome);
    return group;
  }

  private createBuildMesh(row: number, col: number, height: number): THREE.Object3D | null {
    const x = (col - 2) * TILE_SIZE;
    const z = (row - 2) * TILE_SIZE;

    if (height >= 4) {
      const dome = this.createDome();
      const baseY = TILE_TOP + LEVEL_HEIGHT * 3;
      dome.position.set(x, baseY, z);
      dome.scale.setScalar(0.2);
      dome.userData = { baseY, scaleMode: "uniform" };
      return dome;
    }

    const levelIndex = Math.max(0, height - 1);
    const block = this.createBuildingLevel(levelIndex);
    const baseY = TILE_TOP + LEVEL_HEIGHT / 2 + levelIndex * LEVEL_HEIGHT;
    block.position.set(x, baseY, z);
    block.scale.y = 0.2;
    block.userData = { baseY };
    return block;
  }

  private tileToWorld(row: number, col: number, heightIndex: number): THREE.Vector3 {
    const standHeight = TILE_TOP + Math.min(heightIndex, 4) * LEVEL_HEIGHT + 0.2;
    return new THREE.Vector3((col - 2) * TILE_SIZE, standHeight, (row - 2) * TILE_SIZE);
  }

  private workerKey(worker: StateSummary["workers"][number]): string {
    return `${worker.player}-${worker.id}`;
  }

  private findWorkerGroup(key: string): THREE.Group | null {
    return (this.workersGroup.children.find(
      (child) => child.userData?.key === key
    ) as THREE.Group) ?? null;
  }

  private animateFor(durationMs: number, update: (t: number) => void): Promise<void> {
    return new Promise((resolve) => {
      const start = performance.now();
      const tick = (now: number) => {
        const t = Math.min((now - start) / durationMs, 1);
        update(t);
        if (t < 1) {
          requestAnimationFrame(tick);
        } else {
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  private easeOutCubic(t: number): number {
    return 1 - Math.pow(1 - t, 3);
  }

  private clearGroup(group: THREE.Group): void {
    while (group.children.length > 0) {
      const child = group.children[0];
      if (!child) {
        continue;
      }
      group.remove(child);
      child.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const material = object.material;
          if (Array.isArray(material)) {
            material.forEach((mat) => mat.dispose());
          } else {
            material.dispose();
          }
        }
      });
    }
  }
}
