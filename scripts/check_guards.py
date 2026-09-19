"""Local-browser freshness/execution regressions. No model calls or external websites."""

from urllib.parse import quote

from fast_browser_use.browser import Browser, StalePage

HTML = """<!doctype html><title>Guard checks</title>
<style>body{margin:30px}button{width:180px;height:50px}#outside{position:absolute;top:3000px}</style>
<p id="context">Cart total: $10</p>
<button id="target" onclick="window.clicks=(window.clicks||0)+1">Continue</button>
<label>City<input id="field" value="Zurich"></label>
<label><input id="toggle" type="checkbox">Refundable</label>
<select aria-label="Category"><option>All</option><option>Design</option></select>
<p id="outside">Unrelated offscreen text</p>"""


def main():
    browser = Browser("data:text/html," + quote(HTML))
    passed = []
    try:
        page = browser.observe(screenshot=False)
        assert any(a["id"] == "scroll_down" for a in page["actions"])
        browser.evaluate("document.body.style.overflow='hidden'")
        assert not any(a["kind"] == "scroll" for a in browser.observe(screenshot=False)["actions"])
        browser.evaluate("document.body.style.overflow=''")
        page = browser.observe(screenshot=False)
        passed.append("locked document does not offer ineffective page scrolling")
        action = next(a for a in page["actions"] if a["label"] == "Continue")
        browser.evaluate("document.querySelector('#target').style.transform='translateX(200px)'")
        assert browser.fresh(page, action), "Movement should use fresh geometry, not another model call"
        browser.act(action, page)
        assert browser.evaluate("window.clicks") == 1
        passed.append("moving target clicked at its current location")
        page = browser.observe(screenshot=False)

        browser.evaluate("document.querySelector('#outside').textContent='Updated outside the viewport'")
        assert browser.fresh(page)
        passed.append("unrelated offscreen text does not invalidate")

        mutations = {
            "visible context": "document.querySelector('#context').textContent='Cart total: $100'",
            "accessible label": "document.querySelector('#target').setAttribute('aria-label','Delete account')",
            "field property": "document.querySelector('#field').value='London'",
            "checkbox property": "document.querySelector('#toggle').checked=true",
            "disabled target": "document.querySelector('#target').disabled=true",
            "read-only field": "document.querySelector('#field').readOnly=true",
            "hidden target": "document.querySelector('#target').style.display='none'",
            "replaced node": "document.querySelector('#target').outerHTML=document.querySelector('#target').outerHTML",
            "dropdown option": "document.querySelector('select').options[1].text='Coastal'",
        }
        for label, expression in mutations.items():
            browser.evaluate(
                "document.querySelector('#target').style.display='block'; "
                "document.querySelector('#target').disabled=false"
            )
            page = browser.observe(screenshot=False)
            browser.evaluate(expression)
            assert not browser.fresh(page), label
            passed.append(label + " invalidates")

        browser.evaluate(
            "document.querySelector('#target').disabled=false; document.querySelector('#target').style.display='block'"
        )
        page = browser.observe(screenshot=False)
        action = next(a for a in page["actions"] if a["label"] == "Delete account")
        # A textless overlay does not alter the model's semantic state, but must block a click.
        browser.evaluate(
            "const cover=document.createElement('div'); "
            "cover.style.cssText='position:fixed;inset:0;z-index:9999;background:white'; "
            "document.body.append(cover)"
        )
        assert not browser.fresh(page)
        try:
            browser.act(action, page)
        except (RuntimeError, StalePage):
            pass
        else:
            raise AssertionError("Covered target was clicked")
        assert browser.evaluate("window.clicks") == 1
        passed.append("overlay blocked before input")

        browser.evaluate(
            "document.body.innerHTML="
            + repr("""
          <form><p id="price">Total $10</p>
          <button type="button" id="buy">Buy</button>
          <label>Search <input id="query" role="combobox" aria-controls="suggestions"></label>
          <div role="listbox" id="suggestions"></div>
          <label><input id="check" type="checkbox">Enabled</label>
          <label><input id="radio" type="radio">Choice</label>
          <input id="readonly" aria-label="Read only" readonly>
          <input id="secret" type="password" value="never expose this">
          <button id="off" disabled>Disabled</button>
          <select id="category" aria-label="Category">
            <option>All</option><option>Design</option><option disabled>Unavailable</option>
          </select></form><aside id="unrelated">News</aside>
        """)
        )
        page = browser.observe(screenshot=False)
        buy = next(a for a in page["actions"] if a["label"] == "Buy")
        search_field = next(a for a in page["actions"] if a["kind"] == "fill")
        browser.evaluate("document.querySelector('#unrelated').textContent='New unrelated news'")
        assert browser.fresh(page, buy)
        assert browser.fresh(page, search_field)
        assert not browser.fresh(page)
        passed.append("click and fill guards accept unrelated visible updates; terminal guard rejects them")
        for label, expression in {
            "nearby price": "document.querySelector('#price').textContent='Total $100'",
            "form value": "document.querySelector('#query').value='changed'",
            "form toggle": "document.querySelector('#check').checked=true",
            "target replacement": "document.querySelector('#buy').outerHTML=document.querySelector('#buy').outerHTML",
        }.items():
            page = browser.observe(screenshot=False)
            buy = next(a for a in page["actions"] if a["label"] == "Buy")
            browser.evaluate(expression)
            assert not browser.fresh(page, buy), label
            passed.append(label + " invalidates action-specific guard")

        page = browser.observe(screenshot=False)
        actions = page["actions"]
        for role in ("checkbox", "radio"):
            assert {a["kind"] for a in actions if a.get("role") == role} == {"click"}
        assert {a["kind"] for a in actions if a["label"] == "Read only"} == {"click"}
        assert not any(a["label"] == "Disabled" or a.get("value") == "never expose this" for a in actions)
        assert [a["value"] for a in actions if a["kind"] == "select"] == ["Design"]
        passed.append("native controls expose only supported operations and safe values")

        select = next(a for a in actions if a["kind"] == "select")
        browser.act(select, page)
        assert browser.evaluate("document.querySelector('#category').value") == "Design"
        passed.append("native dropdown selects an observed option")

        browser.evaluate(
            "document.querySelector('#query').addEventListener('input',()=>setTimeout(()=>{"
            "document.querySelector('#suggestions').innerHTML='<div role=option>Generated</div>'"
            "},60))"
        )
        page = browser.observe(screenshot=False)
        field = next(a for a in page["actions"] if a["kind"] == "fill")
        browser.act(field, page, text="Generated")
        page = browser.observe(screenshot=False)
        value = browser.evaluate("document.querySelector('#query').value")
        assert value == "Generated", repr(value)
        assert any(a.get("role") == "option" for a in page["actions"])
        passed.append("real text input waits for asynchronous combobox suggestions")

        browser.evaluate(
            "document.body.innerHTML=" + repr('''<input id="later" type="search" aria-label="Query">
            <div id="suggestions"></div><span aria-hidden="true" id="clock"></span>''')
        )
        browser.evaluate("""(() => {
          window.inputs=0;
          document.querySelector('#later').addEventListener('input',()=>{
            window.inputs++;
            setTimeout(()=>{document.querySelector('#suggestions').innerHTML='<button>Suggestion</button>'},180);
          });
          window.clockTimer=setInterval(()=>{document.querySelector('#clock').textContent=Date.now()},20);
        })()""")
        page = browser.observe(screenshot=False)
        field = next(a for a in page["actions"] if a["kind"] == "fill")
        browser.act(field, page, text="Query")
        page = browser.prepare(browser.observe(screenshot=False))
        assert any(a["label"] == "Suggestion" for a in page["actions"])
        assert page["preparation"]["settled"]
        assert browser.evaluate("window.inputs") == 1
        browser.evaluate("clearInterval(window.clockTimer)")
        passed.append("settling observes debounced searchboxes without repeating input or waiting on hidden clocks")

        browser.evaluate("""(() => {
          document.body.innerHTML='<p id="ticker">0</p>';
          window.ticks=0;
          window.tickerTimer=setInterval(()=>{document.querySelector('#ticker').textContent=++window.ticks},20);
        })()""")
        page = browser.prepare(browser.observe(screenshot=False))
        assert not page["preparation"]["settled"]
        assert page["preparation"]["samples"] > 1
        browser.evaluate("clearInterval(window.tickerTimer)")
        passed.append("continuously changing visible content cannot cause an unbounded settle wait")
        browser.call("Page.navigate", url="about:blank")
        assert not browser.fresh(page, field)
        passed.append("navigation invalidates the old document")
        browser.page.wait_for_load_state("domcontentloaded")
        browser.evaluate(
            "document.body.innerHTML='<div role=gridcell aria-label=Day>"
            '<input type=checkbox aria-label="Choose day"></div>\''
        )
        page = browser.observe(screenshot=False)
        assert [a.get("role") for a in page["actions"] if a.get("node")] == ["checkbox"]
        passed.append("calendar cells with interactive children expose only the actual control")
        browser.evaluate(
            "document.body.innerHTML=" + repr('''<input id="styled" type="checkbox" aria-label="Digest subscription"
            style="opacity:0;position:absolute;left:-1000px;width:1px;height:1px">
            <label for="styled">4 stars</label>''')
        )
        page = browser.observe(screenshot=False)
        proxy = next(a for a in page["actions"] if a.get("role") == "checkbox")
        assert proxy["label"] == "Digest subscription" and proxy["checked"] == "false"
        browser.act(proxy, page)
        assert browser.evaluate("document.querySelector('#styled').checked") is True
        page = browser.observe(screenshot=False)
        proxy = next(a for a in page["actions"] if a.get("role") == "checkbox")
        browser.evaluate("document.querySelector('#styled').disabled=true")
        assert not browser.fresh(page, proxy)
        assert not any(a.get("role") == "checkbox" for a in browser.observe(screenshot=False)["actions"])
        passed.append("visible labels safely operate styled checkboxes and track the associated control state")
        browser.evaluate(
            "document.body.style.overflow='hidden'; document.body.innerHTML=" + repr('''
            <aside aria-label="Filter options" style="height:200px;width:250px;overflow-y:auto">
            <div style="height:900px">Filter list</div><button>Last filter</button></aside>''')
        )
        page = browser.observe(screenshot=False)
        scroll = next(a for a in page["actions"] if a["kind"] == "scroll")
        assert scroll["label"] == "Scroll down inside Filter options"
        browser.act(scroll, page)
        browser.page.wait_for_timeout(100)
        assert browser.evaluate("document.querySelector('aside').scrollTop") > 0
        assert browser.evaluate("scrollY") == 0
        assert not browser.fresh(page)
        passed.append("observed nested scrolling moves its own container and invalidates the old state")
    finally:
        browser.close()
    print("\n".join(passed))
    print(f"PASS: {len(passed)} browser guard checks; no model calls")


if __name__ == "__main__":
    main()
