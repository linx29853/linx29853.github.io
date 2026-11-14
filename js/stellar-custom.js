(() => {
  const brand=document.querySelector('#leftbar-region .brand-wrap');
  if(brand && !brand.querySelector('.personal-avatar')) {
    const avatar=document.createElement('a');avatar.className='personal-avatar';avatar.href='/personal/';avatar.setAttribute('aria-label','进入个人页面');brand.prepend(avatar);
  }
  // Stellar 的导航使用扁平菜单；用原生 details 添加可键盘操作的个人子菜单。
  const menu = document.querySelector('#leftbar-region a[href="/link/"]')?.parentElement;
  if (menu) {
    const travelIcon=menu.querySelector('a[title="旅行"] .ui-icon');
    if (travelIcon) travelIcon.innerHTML='<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 5c0-2.4 2.4-3.5 7-3.5s7 1.1 7 3.5v9c0 2-1.4 3.5-3.3 3.8l1.7 2.3c.3.4.2.9-.2 1.1-.4.3-.9.2-1.2-.2l-1-1.4H9l-1 1.4c-.3.4-.8.5-1.2.2s-.5-.7-.2-1.1l1.7-2.3C6.4 17.5 5 16 5 14Z" fill="currentColor" opacity=".5"/><path d="M7.5 6.2c0-1 .8-1.7 1.8-1.7h5.4c1 0 1.8.7 1.8 1.7v2.5c0 .7-.6 1.3-1.3 1.3H8.8c-.7 0-1.3-.6-1.3-1.3Z" fill="currentColor"/><circle cx="8.7" cy="13.7" r="1.1" fill="currentColor"/><circle cx="15.3" cy="13.7" r="1.1" fill="currentColor"/></svg>';
    const icon = {
      person:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="7.5" r="4" fill="currentColor"/><path d="M4 19c0-3.4 3.6-5.4 8-5.4s8 2 8 5.4c0 1.7-1.5 2-3.2 2H7.2C5.5 21 4 20.7 4 19Z" fill="currentColor" opacity=".5"/></svg>',
      music:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 3.5v12.2a3.6 3.6 0 1 1-1.8-3.12V5.1l7-1.6v9.7a3.6 3.6 0 1 1-1.8-3.12V3.5Z" fill="currentColor"/><path d="M13.2 5.1 20.2 3.5v3l-7 1.6Z" fill="currentColor" opacity=".5"/></svg>',
      video:'<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="7" width="18" height="13" rx="4" fill="currentColor" opacity=".5"/><path d="m8 3 3.2 3M16 3l-3.2 3" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><path d="m10 10 5 3-5 3Z" fill="currentColor"/></svg>',
      games:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.2 7h9.6c2.1 0 3.4 1.3 3.9 3.4l.9 4.1c.7 3.3-1.8 5.1-4.2 3.2l-2-1.5H8.6l-2 1.5c-2.4 1.9-4.9.1-4.2-3.2l.9-4.1C3.8 8.3 5.1 7 7.2 7Z" fill="currentColor" opacity=".5"/><path d="M7.5 10v4M5.5 12h4" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/><circle cx="16.5" cy="11" r="1" fill="currentColor"/><circle cx="18.5" cy="13" r="1" fill="currentColor"/></svg>'
    };
    const group=document.createElement('details');group.className='personal-menu';
    if (/^\/(personal|music|bilibili|games)\//.test(location.pathname)) group.open=true;
    group.innerHTML=`<summary class="ui-collection__item ui-interactive personal-menu__summary"><span class="ui-icon">${icon.person}</span><span class="ui-collection__content"><span class="ui-collection__title">个人</span></span><span class="personal-menu__chevron" aria-hidden="true">⌄</span></summary><div class="personal-menu__children"><a href="/music/" class="ui-collection__item ui-interactive"><span class="ui-icon">${icon.music}</span><span class="ui-collection__content"><span class="ui-collection__title">音乐</span></span></a><a href="/bilibili/" class="ui-collection__item ui-interactive"><span class="ui-icon">${icon.video}</span><span class="ui-collection__content"><span class="ui-collection__title">B站空间</span></span></a><a href="/games/" class="ui-collection__item ui-interactive"><span class="ui-icon">${icon.games}</span><span class="ui-collection__content"><span class="ui-collection__title">游戏</span></span></a></div>`;
    const friend=menu.querySelector('a[href="/link/"]');if(friend)friend.before(group);else menu.append(group);
  }
  const archivedHomeUrls = new Set(["/2026/09/08/2021-MRACon-%E7%B2%BE%E8%AF%BB%E8%AE%B2%E8%A7%A3%E6%8A%A5%E5%91%8A/", "/2026/08/30/%E7%BA%BF%E6%80%A7%E7%B3%BB%E7%BB%9F%E5%9F%BA%E7%A1%80%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/", "/2026/08/29/%E5%90%8C%E7%B1%BB%E6%96%87%E7%8C%AE%E5%B9%BF%E6%B3%9B%E6%A3%80%E7%B4%A2%E6%8A%A5%E5%91%8A-MRACon-MRASCon%E4%B8%8E%E5%88%86%E5%B8%83%E5%BC%8F%E8%87%AA%E9%80%82%E5%BA%94%E6%8E%A7%E5%88%B6/", "/2026/08/29/Popov%E8%B6%85%E7%A8%B3%E5%AE%9A%E6%96%B9%E6%B3%95%E4%B8%8EMRAC-%E7%B2%BE%E7%AE%80%E7%AC%94%E8%AE%B0/", "/2026/08/27/MRASC_leader_following_consensus_%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/", "/2026/08/26/%E6%9D%8E%E9%9B%85%E6%99%AE%E8%AF%BA%E5%A4%AB%E7%A8%B3%E5%AE%9A%E6%80%A7%E4%B8%8E%E7%BA%BF%E6%80%A7%E7%B3%BB%E7%BB%9F%E6%96%B9%E6%B3%95-%E7%B2%BE%E7%AE%80%E7%AC%94%E8%AE%B0/", "/2026/08/26/MRAC%E6%95%B0%E5%AD%A6%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E7%82%B9%E7%AC%94%E8%AE%B0/"]);
  if (location.pathname === "/" || /^\/page\/\d+\/?$/.test(location.pathname)) {
    document.querySelectorAll('#leftbar-region [data-density="compact"] a[href]').forEach(a => {
      if (archivedHomeUrls.has(a.getAttribute("href"))) a.remove();
    });
  }
  const home=location.pathname==='/';
  {
    const target=(home && document.querySelector('.post-list')) || document.querySelector('main');
    if(target){
      let line;
      {
        if(home) target.id='latest-articles';
        const hero=document.createElement('section');hero.className='home-hero';
        hero.innerHTML=`<video class="home-hero__video" autoplay muted loop playsinline preload="metadata" aria-hidden="true"><source src="/video/home-background.mp4" type="video/mp4"></video><div class="home-hero__shade"></div><div class="home-hero__content"><div class="home-hero__intro"><h1>LenoVo的小站</h1><p class="home-hero__poem" aria-live="polite"></p></div><a class="home-hero__scroll" href="${home ? '#latest-articles' : '#main'}" aria-label="向下浏览页面内容">⌄</a></div>`;
        document.body.prepend(hero);line=hero.querySelector('.home-hero__poem');
        // 进入栏目时从内容开始，向上滚动即可看到首页画面。
        if(!home && !location.hash) window.scrollTo({top:hero.offsetHeight,behavior:'instant'});
      }
      const script=document.createElement('script');script.src='https://sdk.jinrishici.com/v2/browser/jinrishici.js';
      script.onload=()=>window.jinrishici.load(result=>{
        const poem=result?.data?.content;if(!poem){line.remove();return;}
        let index=0,deleting=false;
        function type(){line.textContent=poem.slice(0,index);index+=deleting?-1:1;
          if(index>poem.length){deleting=true;setTimeout(type,3000);return;}
          if(index<0){index=0;deleting=false;setTimeout(type,700);return;}
          setTimeout(type,deleting?55:130);}
        type();
      });
      script.onerror=()=>line.remove();document.head.appendChild(script);
    }
  }
})();
