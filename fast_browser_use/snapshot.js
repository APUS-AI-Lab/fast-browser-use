(() => {
  if (!document.body) return null;
  const cache = window.__fastBrowserUse ||= {ids:new WeakMap(), nodes:new Map(), next:1};
  const identity = e => {
    if (!cache.ids.has(e)) cache.ids.set(e,cache.next++);
    const id=cache.ids.get(e); cache.nodes.set(id,e); return id;
  };
  for (const [id,e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);
  const safe = e => !['password','file','hidden'].includes(e.type);
  const visible = e => !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
  const name = (e,seen=new Set()) => {
    if (!e || seen.has(e)) return '';
    seen.add(e);
    if (e.tagName==='LABEL' && e.control && ['checkbox','radio'].includes(e.control.type)) {
      const controlName=name(e.control,seen);
      if (controlName) return controlName;
    }
    const referenced=(e.getAttribute('aria-labelledby')||'').split(/\s+/)
      .map(id=>name(document.getElementById(id),seen)).filter(Boolean).join(' ');
    return referenced || e.getAttribute('aria-label') ||
      [...(e.labels||[])].map(l=>name(l,seen)).filter(Boolean).join(' ') ||
      (['button','submit','reset'].includes(e.type) ? e.value : '') || e.getAttribute('alt') ||
      (e.tagName==='INPUT' ? '' : [...e.childNodes].map(n=>n.nodeType===3 ? n.textContent :
        n.nodeType===1 && n.getAttribute('aria-hidden')!=='true' ? name(n,seen) : '').join(' ').trim()) ||
      e.getAttribute('title') || e.getAttribute('placeholder') || '';
  };
  const roles=['button','link','checkbox','radio','switch','tab','menuitem','menuitemradio',
    'option','gridcell','combobox','textbox','searchbox','spinbutton'];
  const selector='a[href],button,input,textarea,select,summary,label,[contenteditable="true"],'+
    roles.map(role=>'[role="'+role+'"]').join(',');
  const role = e => {
    const explicit=e.getAttribute('role');
    if (roles.includes(explicit)) return explicit;
    if (e.tagName==='LABEL' && e.control && ['checkbox','radio'].includes(e.control.type)) return e.control.type;
    if (e.tagName==='BUTTON' || e.tagName==='SUMMARY') return 'button';
    if (e.tagName==='A') return 'link';
    if (e.tagName==='SELECT') return 'combobox';
    if (e.tagName==='TEXTAREA' || e.isContentEditable) return 'textbox';
    if (e.tagName==='INPUT') {
      if (['checkbox','radio'].includes(e.type)) return e.type;
      if (['button','submit','reset','image'].includes(e.type)) return 'button';
      if (e.type==='search') return 'searchbox';
      if (e.type==='number') return 'spinbutton';
      if (['text','email','url','tel'].includes(e.type)) return 'textbox';
    }
    return null;
  };
  cache.pageKey=()=>[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    [...document.querySelectorAll('input,textarea,select')].filter(safe)
      .map(e=>[identity(e),e.value,e.checked,e.selectedIndex,e.disabled,e.readOnly])];
  cache.guard=e=>{
    if (!e?.isConnected || !visible(e)) return null;
    const control=e.tagName==='LABEL' ? e.control : e;
    const scope=e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [identity(e),role(e),name(e),control?.value??null,control?.checked??null,control?.selectedIndex??null,
      control?.readOnly??null,!!control?.matches(':disabled'),control?.getAttribute('aria-disabled'),
      e.getAttribute('aria-expanded'),e.getAttribute('aria-checked'),e.getAttribute('aria-selected'),
      e.getAttribute('href'),scope?.innerText?.slice(0,6000)||''];
  };
  const actions=[];
  for (const e of document.querySelectorAll(selector)) {
    if (!safe(e) || !visible(e) || e.matches(':disabled') || e.closest('[aria-disabled="true"]')) continue;
    const control=e.tagName==='LABEL' ? e.control : e;
    if (e.tagName==='LABEL') {
      if (!control || !['checkbox','radio'].includes(control.type) || control.matches(':disabled') ||
          control.closest('[aria-disabled="true"]')) continue;
      const c=control.getBoundingClientRect(), cx=c.x+c.width/2, cy=c.y+c.height/2;
      if (visible(control) && c.width>0 && c.height>0 && cx>=0 && cy>=0 && cx<innerWidth && cy<innerHeight &&
          control.contains(document.elementFromPoint(cx,cy))) continue;
    }
    const r=e.getBoundingClientRect(), x=r.x+r.width/2, y=r.y+r.height/2, rname=role(e);
    if (!rname || r.width<=0 || r.height<=0 || x<0 || y<0 || x>=innerWidth || y>=innerHeight) continue;
    if (!e.contains(document.elementFromPoint(x,y))) continue;
    if (rname==='gridcell' && e.querySelector('button,input,[role="button"],[role="checkbox"],[role="radio"]')) continue;
    const base={node:identity(e),role:rname,label:name(e)||rname,
      rect:{x:r.x,y:r.y,w:r.width,h:r.height}};
    for (const key of ['checked','selected','expanded']) {
      const value=e.getAttribute('aria-'+key);
      if (value!==null) base[key]=value;
    }
    if (['checkbox','radio'].includes(control?.type)) base.checked=String(control.checked);
    if (e.tagName==='SELECT') {
      for (const o of e.options) if (!o.selected && !o.disabled && !o.closest('optgroup[disabled]'))
        actions.push({...base,kind:'select',value:o.value,
          current_value:[...e.selectedOptions].map(o=>o.label).join(', '),label:base.label+' → '+o.label});
    } else {
      const editable=!e.readOnly && e.getAttribute('aria-readonly')!=='true' &&
        (['textbox','searchbox','spinbutton'].includes(rname) ||
          (rname==='combobox' && ['INPUT','TEXTAREA'].includes(e.tagName)));
      const value='value' in e ? String(e.value) :
        e.isContentEditable || rname==='combobox' ? e.innerText.trim() : '';
      actions.push({...base,kind:editable?'fill':'click',value});
      if (editable) actions.push({...base,kind:'click',value,label:'Open '+base.label});
    }
  }
  const words=[], walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  const range=document.createRange(); let node,length=0;
  while ((node=walker.nextNode()) && length<6000) {
    const value=node.textContent.trim(), parent=node.parentElement;
    if (!value || !parent || parent.closest('script,style,noscript,template') || !visible(parent)) continue;
    range.selectNodeContents(node); const r=range.getBoundingClientRect();
    if (r.width>0 && r.height>0 && r.bottom>0 && r.top<innerHeight && r.right>0 && r.left<innerWidth) {
      const x=(Math.max(0,r.left)+Math.min(innerWidth,r.right))/2;
      const y=(Math.max(0,r.top)+Math.min(innerHeight,r.bottom))/2;
      if (parent.contains(document.elementFromPoint(x,y))) { words.push(value); length+=value.length; }
    }
  }
  const text=words.join('\n').slice(0,6000), height=document.documentElement.scrollHeight;
  const dialogs=[...document.querySelectorAll('dialog[open],[role="dialog"],[aria-modal="true"]')]
    .filter(e=>{
      if (!visible(e)) return false;
      const r=e.getBoundingClientRect(), left=Math.max(0,r.left), right=Math.min(innerWidth,r.right);
      const top=Math.max(0,r.top), bottom=Math.min(innerHeight,r.bottom);
      return right>left && bottom>top && e.contains(document.elementFromPoint((left+right)/2,(top+bottom)/2));
    }).map(e=>({label:name(e).slice(0,160),modal:e.getAttribute('aria-modal')==='true'}));
  const rootOverflow=getComputedStyle(document.documentElement).overflowY;
  const bodyStyle=getComputedStyle(document.body);
  const pageScroll=!['hidden','clip'].includes(rootOverflow) &&
    !['hidden','clip'].includes(bodyStyle.overflowY) && bodyStyle.position!=='fixed' && !dialogs.some(d=>d.modal);
  const root=document.scrollingElement;
  const scrollable=e=>['auto','scroll'].includes(getComputedStyle(e).overflowY) && e.scrollHeight>e.clientHeight+2;
  cache.scrollGuard=e=>e?.isConnected ? [identity(e),e.scrollTop,e.scrollHeight,e.clientHeight] : null;
  cache.scrollPoint=e=>{
    if (!e?.isConnected) return null;
    const r=e===root ? {left:0,top:0,right:innerWidth,bottom:innerHeight} : e.getBoundingClientRect();
    const l=Math.max(0,r.left), t=Math.max(0,r.top), w=Math.min(innerWidth,r.right)-l, h=Math.min(innerHeight,r.bottom)-t;
    if (w<=0 || h<=0) return null;
    for (const [fx,fy] of [[.5,.5],[.1,.5],[.9,.5],[.5,.1],[.5,.9]]) {
      const x=l+w*fx,y=t+h*fy; let owner=document.elementFromPoint(x,y);
      while (owner && owner!==root && !scrollable(owner)) owner=owner.parentElement;
      if (owner===e) return {x,y};
    }
    return null;
  };
  const scrolling=[];
  for (const e of document.querySelectorAll('body *')) {
    if (e===root || !visible(e) || !scrollable(e) || !cache.scrollPoint(e)) continue;
    const label=(e.getAttribute('aria-label') || e.querySelector('h1,h2,h3,h4')?.innerText || e.innerText).trim().slice(0,100);
    for (const delta of [560,-560]) {
      if (delta>0 ? e.scrollTop+e.clientHeight>=e.scrollHeight-2 : e.scrollTop<=0) continue;
      scrolling.push({id:'scroll_'+(delta>0?'down':'up')+'_'+identity(e),kind:'scroll',node:identity(e),
        label:'Scroll '+(delta>0?'down':'up')+' inside '+(label||'scrollable region'),delta,
        scroll_state:cache.scrollGuard(e)});
    }
  }
  const pagePoint=pageScroll && cache.scrollPoint(root);
  const page_key=cache.pageKey(), guards={};
  for (const a of actions) if (!(a.node in guards)) guards[a.node]=cache.guard(cache.nodes.get(a.node));
  // Compare meaning and identity. Geometry is always resolved and hit-tested just before input.
  const semantics=actions.map(({rect,...action})=>action);
  const marker=[performance.timeOrigin,location.href,scrollX,scrollY,innerWidth,innerHeight,
    document.title,text,semantics,page_key[6],document.documentElement.lang,pageScroll,dialogs,scrolling];
  const omitted_actions=Math.max(0,actions.length-250);
  actions.splice(250);
  actions.forEach((a,i)=>a.id='e'+(i+1));
  actions.push(...scrolling);
  if (pagePoint && scrollY+innerHeight<height-2) actions.push({id:'scroll_down',kind:'scroll',label:'Scroll down',delta:560});
  if (pagePoint && scrollY>0) actions.push({id:'scroll_up',kind:'scroll',label:'Scroll up',delta:-560});
  actions.push({id:'wait',kind:'wait',label:'Wait for the page to update'});
  const ready=document.readyState!=='loading' && [...document.querySelectorAll('link[rel="stylesheet"]')]
    .every(e=>e.disabled || e.sheet || !matchMedia(e.media||'all').matches);
  return {url:location.href,title:document.title,language:document.documentElement.lang,ready,dialogs,w:innerWidth,h:innerHeight,text,
    scroll:{y:scrollY,height},actions,marker,page_key,guards,omitted_actions};
})()
