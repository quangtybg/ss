
(() => {
  const canvas = document.getElementById('game');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const COLS = 10, ROWS = 20, BLOCK = Math.floor(W / COLS);
  const GRID_H = BLOCK * ROWS;
  const COLORS = {
    I:'#50c8ff', O:'#f0e65a', T:'#c878fa', S:'#68dc68', Z:'#f07878', J:'#788cf8', L:'#f0aa46',
    G:'#8c8c98'
  };
  const SHAPES = {
    I:[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    O:[[0,1,1,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]],
    T:[[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
    S:[[0,1,1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]],
    Z:[[1,1,0,0],[0,1,1,0],[0,0,0,0],[0,0,0,0]],
    J:[[1,0,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
    L:[[0,0,1,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]],
  };
  const PIECES = Object.keys(SHAPES);

  const scoreEl = document.getElementById('score');
  const levelEl = document.getElementById('level');
  const linesEl = document.getElementById('lines');
  const nameInput = document.getElementById('name');
  const startBtn = document.getElementById('startBtn');
  const statusEl = document.getElementById('status');

  let grid = [];
  let bag = [];
  let queue = [];
  let active = null;
  let hold = null, holdUsed = false;
  let score = 0, lines = 0, level = 0;
  let gravity = 0.8; // seconds per row at level 0
  let dropAcc = 0;
  let paused = false;
  let running = false;

  function reset(){
    grid = [...Array(ROWS)].map(()=>Array(COLS).fill(null));
    bag = []; queue = []; active = null; hold = null; holdUsed=false;
    score=0; lines=0; level=0; gravity=0.8; dropAcc=0; paused=false;
    spawn();
    updateHUD();
  }

  function refillBag(){
    bag = PIECES.slice();
    for(let i=bag.length-1;i>0;i--){const j=(Math.random()* (i+1))|0; [bag[i],bag[j]]=[bag[j],bag[i]];}
  }
  function nextPiece(){
    if(!bag.length) refillBag();
    return new Piece(bag.pop());
  }
  function ensureQueue(){
    while(queue.length<5) queue.push(nextPiece());
  }
  function spawn(){
    ensureQueue();
    active = queue.shift();
    holdUsed = false;
    for(let k=0;k<2;k++){
      if(!valid(active.cells())) active.origin[0]++;
    }
    if(!valid(active.cells())){
      // game over
      running = false;
      postScore();
      statusEl.textContent = 'Game Over! Điểm đã được lưu.';
    }
  }

  class Piece{
    constructor(id){
      this.id = id;
      this.m = SHAPES[id].map(r=>r.slice());
      this.origin = [0, COLS/2-2|0];
      this.onGround = 0;
      this.lockDelay = 0.5;
    }
    clone(){
      const p = new Piece(this.id);
      p.m = this.m.map(r=>r.slice());
      p.origin = this.origin.slice();
      p.onGround = this.onGround;
      return p;
    }
    cells(){
      const out=[]; const [r0,c0]=this.origin;
      for(let r=0;r<4;r++)for(let c=0;c<4;c++) if(this.m[r][c]) out.push([r0+r, c0+c]);
      return out;
    }
  }

  function rotateCW(m){
    const n= m.length; const res=[...Array(n)].map(()=>Array(n).fill(0));
    for(let r=0;r<n;r++)for(let c=0;c<n;c++) res[c][n-1-r] = m[r][c];
    return res;
  }
  function rotateCCW(m){
    const n= m.length; const res=[...Array(n)].map(()=>Array(n).fill(0));
    for(let r=0;r<n;r++)for(let c=0;c<n;c++) res[n-1-c][r] = m[r][c];
    return res;
  }
  function valid(cells){
    for(const [r,c] of cells){
      if(r<0||r>=ROWS||c<0||c>=COLS) return false;
      if(grid[r][c]) return false;
    }
    return true;
  }

  function move(dx){
    const t = active.clone(); t.origin[1]+=dx;
    if(valid(t.cells())) active.origin=t.origin;
  }
  function softDrop(){
    const t = active.clone(); t.origin[0]+=1;
    if(valid(t.cells())){ active.origin=t.origin; score+=1; return true; }
    return false;
  }
  function hardDrop(){
    let d=0; while(softDrop()) d++;
    score += d*2;
    lockPiece();
  }
  function canFall(){
    const t=active.clone(); t.origin[0]+=1; return valid(t.cells());
  }
  function tick(dt){
    if(!running || paused) return;
    dropAcc += dt;
    if(canFall()){
      if(dropAcc >= gravity){
        dropAcc = 0;
        active.origin[0]++;
      }
    } else {
      active.onGround += dt;
      if(active.onGround >= active.lockDelay) lockPiece();
    }
  }
  function lockPiece(){
    for(const [r,c] of active.cells()){
      if(r>=0 && r<ROWS && c>=0 && c<COLS) grid[r][c] = active.id;
    }
    const cleared = clearLines();
    if(cleared>0){
      lines += cleared;
      score += [0,100,300,500,800][cleared] * (level+1);
      const nl = (lines/10)|0;
      if(nl>level){ level=nl; gravity = Math.max(0.07, 0.8 - level*0.08); }
    }
    spawn();
    updateHUD();
  }
  function clearLines(){
    let nr = grid.filter(row => row.some(x=>x===null));
    const cleared = ROWS - nr.length;
    while(nr.length<ROWS) nr.unshift(Array(COLS).fill(null));
    grid = nr; return cleared;
  }

  function wallKick(rotated){
    const test = active.clone(); test.m = rotated;
    const offs = [[0,0],[0,-1],[0,1],[-1,0],[1,0],[0,-2],[0,2]];
    for(const [dr,dc] of offs){
      const t = active.clone(); t.m = rotated; t.origin = [active.origin[0]+dr, active.origin[1]+dc];
      if(valid(t.cells())){ active.m = rotated; active.origin = t.origin; return true; }
    }
    return false;
  }

  function rotate(cw=true){
    if(active.id==='O') return;
    const rotated = cw ? rotateCW(active.m) : rotateCCW(active.m);
    wallKick(rotated);
  }

  // Rendering
  function draw(){
    ctx.clearRect(0,0,W,H);
    // grid background
    ctx.fillStyle = '#0f1020';
    ctx.fillRect(0,0,W,H);

    // locked
    for(let r=0;r<ROWS;r++){
      for(let c=0;c<COLS;c++){
        const id = grid[r][c];
        if(id){
          const x=c*BLOCK, y=r*BLOCK;
          ctx.fillStyle = COLORS[id] || '#999';
          ctx.fillRect(x,y,BLOCK,BLOCK);
          ctx.strokeStyle = '#0b0d18';
          ctx.strokeRect(x,y,BLOCK,BLOCK);
        }
      }
    }
    // ghost
    const g = active.clone();
    while(true){
      g.origin[0]++;
      if(!valid(g.cells())){ g.origin[0]--; break; }
    }
    ctx.strokeStyle = '#9aa';
    for(const [r,c] of g.cells()){
      ctx.strokeRect(c*BLOCK, r*BLOCK, BLOCK, BLOCK);
    }
    // active
    for(const [r,c] of active.cells()){
      const x=c*BLOCK, y=r*BLOCK;
      ctx.fillStyle = COLORS[active.id];
      ctx.fillRect(x,y,BLOCK,BLOCK);
      ctx.strokeStyle = '#0b0d18';
      ctx.strokeRect(x,y,BLOCK,BLOCK);
    }
    requestAnimationFrame(draw);
  }

  // HUD
  function updateHUD(){
    scoreEl.textContent = String(score);
    levelEl.textContent = String(level);
    linesEl.textContent = String(lines);
  }

  // Input
  let dasTimer=0, arrTimer=0, moveHeld=false, lastDir=0;
  const DAS=160, ARR=35;
  const keys = {left:false, right:false, down:false};
  function stepInput(dt){
    if(!running || paused) return;
    if(keys.left || keys.right){
      const dir = keys.left ? -1 : 1;
      if(!moveHeld || dir!==lastDir){ move(dir); moveHeld=true; lastDir=dir; dasTimer=0; arrTimer=0; }
      else {
        dasTimer += dt*1000;
        if(dasTimer >= DAS){
          arrTimer += dt*1000;
          while(arrTimer >= ARR){ move(dir); arrTimer -= ARR; }
        }
      }
    }else{
      moveHeld=false; dasTimer=0; arrTimer=0;
    }
    if(keys.down) softDrop();
  }

  document.addEventListener('keydown', (e)=>{
    if(!running) return;
    if(e.code==='ArrowLeft' || e.code==='KeyA') keys.left=true;
    else if(e.code==='ArrowRight' || e.code==='KeyD') keys.right=true;
    else if(e.code==='ArrowDown' || e.code==='KeyS') keys.down=true;
    else if(e.code==='ArrowUp' || e.code==='KeyX') rotate(true);
    else if(e.code==='KeyZ') rotate(false);
    else if(e.code==='Space') hardDrop();
  });
  document.addEventListener('keyup', (e)=>{
    if(e.code==='ArrowLeft' || e.code==='KeyA') keys.left=false;
    else if(e.code==='ArrowRight' || e.code==='KeyD') keys.right=false;
    else if(e.code==='ArrowDown' || e.code==='KeyS') keys.down=false;
  });

  // Game loop
  let last=performance.now();
  function loop(now){
    const dt = (now-last)/1000; last = now;
    stepInput(dt);
    tick(dt);
    requestAnimationFrame(loop);
  }

  // Score POST
  async function postScore(){
    const name = (nameInput.value||'Anonymous').trim().slice(0,40);
    try{
      await fetch('/api/score', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, score})
      });
    }catch(e){ console.error(e); }
  }

  // Start button
  startBtn.addEventListener('click', ()=>{
    if(running){ statusEl.textContent = 'Đang chơi...'; return; }
    statusEl.textContent = 'Bắt đầu! Chúc vui.';
    reset();
    running = true;
  });

  // init
  reset();
  requestAnimationFrame(draw);
  requestAnimationFrame(loop);
})();
