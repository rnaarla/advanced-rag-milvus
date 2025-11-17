(() => {
  const API = '';
  const $ = (id) => document.getElementById(id);
  let sessionId = localStorage.getItem('session_id') || null;

  const addMessage = (role, text) => {
    const el = document.createElement('div');
    el.className = `msg ${role}`;
    el.textContent = text;
    $('messages').appendChild(el);
    $('messages').scrollTop = $('messages').scrollHeight;
  };

  const renderSuggestions = (list) => {
    const box = $('suggestions');
    box.innerHTML = '';
    if (!list || !list.length) return;
    list.forEach(s => {
      const el = document.createElement('div');
      el.className = 'suggestion';
      el.textContent = s;
      el.onclick = () => sendStream(s);
      box.appendChild(el);
    });
  };

  const loadHistory = async () => {
    if (!sessionId) return;
    const res = await fetch(`${API}/chat/history?session_id=${encodeURIComponent(sessionId)}`);
    const data = await res.json();
    $('messages').innerHTML = '';
    (data.messages || []).forEach(m => addMessage(m.role, m.content));
  };

  const ensureSession = async () => {
    if (sessionId) return sessionId;
    const res = await fetch(`${API}/chat/start`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    const data = await res.json();
    sessionId = data.session_id;
    localStorage.setItem('session_id', sessionId);
    pushSession(sessionId);
    return sessionId;
  };

  const pushSession = (sid) => {
    const node = document.createElement('a');
    node.href = '#';
    node.className = 'chat-link';
    node.textContent = sid;
    node.onclick = async (e) => {
      e.preventDefault();
      sessionId = sid;
      localStorage.setItem('session_id', sid);
      await loadHistory();
    };
    $('sessions').prepend(node);
  };

  const sendStream = async (query) => {
    await ensureSession();
    addMessage('user', query);
    const url = `${API}/chat/stream?session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(query)}`;
    const es = new EventSource(url);
    let buffer = '';
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.token) {
          buffer += data.token;
        }
        if (data.done) {
          if (buffer) addMessage('assistant', buffer);
          renderSuggestions(data.suggestions || []);
          es.close();
        }
      } catch (err) { /* ignore */ }
    };
    es.onerror = () => {
      es.close();
    };
  };

  $('send').onclick = async () => {
    const q = $('input').value.trim();
    if (!q) return;
    $('input').value = '';
    sendStream(q);
  };
  $('input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') $('send').click();
  });

  $('newChat').onclick = async () => {
    localStorage.removeItem('session_id');
    sessionId = null;
    await ensureSession();
    $('messages').innerHTML = '';
  };

  $('clearChat').onclick = async () => {
    if (!sessionId) return;
    await fetch(`${API}/chat/clear`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId }) });
    $('messages').innerHTML = '';
  };

  (async () => {
    if (!sessionId) {
      await ensureSession();
      pushSession(sessionId);
    } else {
      pushSession(sessionId);
      await loadHistory();
    }
  })();
})();


