// Attach the server-issued CSRF token to existing same-origin workspace calls.
(() => {
  const token = document.querySelector('meta[name="csrf-token"]')?.content;
  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, options = {}) => {
    const url = new URL(input instanceof Request ? input.url : input, location.href);
    const method = (options.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    if (url.origin === location.origin && !['GET', 'HEAD', 'OPTIONS'].includes(method)) {
      const headers = new Headers(options.headers || (input instanceof Request ? input.headers : undefined));
      headers.set('X-CSRF-Token', token);
      options = {...options, headers};
    }
    return nativeFetch(input, options);
  };
})();
