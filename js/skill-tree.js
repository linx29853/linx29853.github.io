(() => {
  'use strict';

  const root = document.getElementById('skill-tree-app');
  if (!root) return;

  const storageKey = 'embodied-motion-skill-tree-v1';
  const sectionsEl = document.getElementById('skill-tree-sections');
  const statusEl = document.getElementById('skill-tree-status');
  const emptyEl = document.getElementById('skill-tree-empty');
  const searchEl = document.getElementById('skill-tree-search');
  let sections = [];
  let items = [];
  let saved = { schema: 1, items: {} };
  let filter = 'all';
  let canSave = true;

  function element(tag, className, value) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  }

  function parseMarkdown(markdown) {
    const parsed = [];
    let section = null;
    let group = null;
    for (const line of markdown.split(/\r?\n/)) {
      const sectionMatch = line.match(/^##\s+(.+)$/);
      const groupMatch = line.match(/^###\s+(.+)$/);
      const itemMatch = line.match(/^- \[([ xX])\]\s+(.+)$/);
      if (sectionMatch) {
        section = { title: sectionMatch[1].trim(), groups: [] };
        parsed.push(section);
        group = null;
      } else if (groupMatch && section) {
        group = { title: groupMatch[1].trim(), items: [] };
        section.groups.push(group);
      } else if (itemMatch && section && group) {
        const label = itemMatch[2].trim();
        group.items.push({
          id: `${section.title}::${group.title}::${label}`,
          label,
          defaultDone: itemMatch[1].toLowerCase() === 'x',
          section,
          group
        });
      }
    }
    return parsed.filter(part => part.groups.some(partGroup => partGroup.items.length));
  }

  function loadSaved() {
    try {
      const data = JSON.parse(localStorage.getItem(storageKey) || 'null');
      if (data && data.schema === 1 && data.items && typeof data.items === 'object' && !Array.isArray(data.items)) saved = data;
    } catch (_error) {
      canSave = false;
      statusEl.textContent = '当前浏览器无法保存进度；你仍可勾选，并使用“导出进度”备份。';
    }
  }

  function save() {
    if (!canSave) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(saved));
    } catch (_error) {
      canSave = false;
      statusEl.textContent = '浏览器存储不可用，请使用“导出进度”保存当前记录。';
    }
  }

  function itemState(item) {
    const value = saved.items[item.id];
    return {
      done: value && typeof value.done === 'boolean' ? value.done : item.defaultDone,
      note: value && typeof value.note === 'string' ? value.note : ''
    };
  }

  function setItemState(item, next) {
    saved.items[item.id] = { done: Boolean(next.done), note: String(next.note || '').slice(0, 10000) };
    save();
    updateProgress();
    applyFilter();
  }

  function progress(partItems) {
    return { done: partItems.filter(item => itemState(item).done).length, total: partItems.length };
  }

  function updateProgress() {
    const all = progress(items);
    const percent = all.total ? Math.round(all.done / all.total * 100) : 0;
    document.getElementById('skill-tree-done').textContent = all.done;
    document.getElementById('skill-tree-total').textContent = all.total;
    document.getElementById('skill-tree-percent').textContent = `${percent}%`;
    document.getElementById('skill-tree-progress-fill').style.width = `${percent}%`;
    root.querySelector('.skill-tree-progress-track').setAttribute('aria-valuenow', String(percent));
    for (const section of sections) {
      const values = progress(section.groups.flatMap(group => group.items));
      const sectionPercent = values.total ? values.done / values.total * 100 : 0;
      section.countEl.textContent = `${values.done} / ${values.total}`;
      section.fillEl.style.width = `${sectionPercent}%`;
    }
  }

  function render() {
    sectionsEl.replaceChildren();
    for (const section of sections) {
      const card = element('section', 'skill-tree-section');
      const head = element('div', 'skill-tree-section-head');
      head.append(element('h3', '', section.title));
      section.countEl = element('span', 'skill-tree-section-count');
      head.append(section.countEl);
      card.append(head);
      const track = element('div', 'skill-tree-section-track');
      section.fillEl = element('span');
      track.append(section.fillEl);
      card.append(track);
      section.node = card;

      for (const group of section.groups) {
        const block = element('div', 'skill-tree-group');
        block.append(element('h4', '', group.title));
        group.node = block;
        for (const item of group.items) {
          const row = element('div', 'skill-tree-item');
          const label = element('label', 'skill-tree-item-label');
          const checkbox = element('input');
          checkbox.type = 'checkbox';
          checkbox.checked = itemState(item).done;
          checkbox.setAttribute('aria-label', item.label);
          label.append(checkbox, element('span', '', item.label));
          row.append(label);

          const toggle = element('button', 'skill-tree-note-toggle', itemState(item).note ? '编辑学习记录' : '添加学习记录');
          toggle.type = 'button';
          toggle.setAttribute('aria-expanded', 'false');
          const note = element('textarea', 'skill-tree-note');
          note.value = itemState(item).note;
          note.placeholder = '写下对应笔记、实验链接或尚未解决的问题…';
          note.setAttribute('aria-label', `${item.label}的学习记录`);
          note.hidden = true;
          row.append(toggle, note);
          row.classList.toggle('is-done', checkbox.checked);
          item.node = row;
          item.checkbox = checkbox;
          item.noteEl = note;
          item.toggleEl = toggle;

          checkbox.addEventListener('change', () => {
            row.classList.toggle('is-done', checkbox.checked);
            setItemState(item, { done: checkbox.checked, note: note.value });
          });
          toggle.addEventListener('click', () => {
            note.hidden = !note.hidden;
            toggle.setAttribute('aria-expanded', String(!note.hidden));
            if (!note.hidden) note.focus();
          });
          note.addEventListener('input', () => {
            toggle.textContent = note.value ? '编辑学习记录' : '添加学习记录';
            setItemState(item, { done: checkbox.checked, note: note.value });
          });
          block.append(row);
        }
        card.append(block);
      }
      sectionsEl.append(card);
    }
    updateProgress();
    applyFilter();
  }

  function applyFilter() {
    const query = searchEl.value.trim().toLocaleLowerCase();
    let visible = 0;
    for (const section of sections) {
      let sectionVisible = 0;
      for (const group of section.groups) {
        let groupVisible = 0;
        for (const item of group.items) {
          const done = itemState(item).done;
          const matchesText = !query || `${section.title} ${group.title} ${item.label}`.toLocaleLowerCase().includes(query);
          const matchesStatus = filter === 'all' || (filter === 'done' ? done : !done);
          const show = matchesText && matchesStatus;
          item.node.hidden = !show;
          if (show) groupVisible++;
        }
        group.node.hidden = groupVisible === 0;
        sectionVisible += groupVisible;
      }
      section.node.hidden = sectionVisible === 0;
      visible += sectionVisible;
    }
    emptyEl.hidden = visible !== 0;
  }

  searchEl.addEventListener('input', applyFilter);
  root.querySelectorAll('[data-filter]').forEach(button => {
    button.addEventListener('click', () => {
      filter = button.dataset.filter;
      root.querySelectorAll('[data-filter]').forEach(option => option.setAttribute('aria-pressed', String(option === button)));
      applyFilter();
    });
  });

  document.getElementById('skill-tree-export').addEventListener('click', () => {
    const snapshot = { schema: 1, exportedAt: new Date().toISOString(), items: {} };
    for (const item of items) snapshot.items[item.id] = itemState(item);
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = element('a');
    link.href = url;
    link.download = `具身智能运动控制技能树进度-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  document.getElementById('skill-tree-import').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      if (file.size > 2000000) throw new Error('文件过大');
      const data = JSON.parse(await file.text());
      if (data.schema !== 1 || !data.items || typeof data.items !== 'object' || Array.isArray(data.items)) throw new Error('文件格式不正确');
      const next = {};
      for (const item of items) {
        const value = data.items[item.id];
        if (value && typeof value.done === 'boolean') {
          next[item.id] = { done: value.done, note: typeof value.note === 'string' ? value.note.slice(0, 10000) : '' };
        }
      }
      saved.items = next;
      save();
      render();
      statusEl.textContent = '进度已导入。';
    } catch (error) {
      statusEl.textContent = `导入失败：${error.message}`;
    }
    event.target.value = '';
  });

  fetch('/skill-tree/skills.txt')
    .then(response => {
      if (!response.ok) throw new Error(`无法读取清单（${response.status}）`);
      return response.text();
    })
    .then(markdown => {
      sections = parseMarkdown(markdown);
      items = sections.flatMap(section => section.groups.flatMap(group => group.items));
      if (!items.length) throw new Error('清单中没有可勾选的技能点');
      loadSaved();
      render();
      if (canSave) statusEl.textContent = '修改会自动保存到当前浏览器。';
    })
    .catch(error => { statusEl.textContent = `技能树加载失败：${error.message}`; });
})();
