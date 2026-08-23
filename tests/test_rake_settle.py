"""Auto-settle after delivery: the request, the guards, and the failure contract.

Settlement is not ingestion. Ingest writes one InspectionRun per camera; settle
then clusters those rows into 7-minute rake windows, assigns `rake_id` and pairs
loaded against empty by shared 11-digit wagon numbers. Until it runs, a freshly
ingested train has rows but no rake grouping -- which is why it was being done
by hand after every run.

Three things carry real risk, and each has tests here:

* **`dry_run` defaults to TRUE in that API.** A request that omits it returns
  200, reports what it *would* change, and writes nothing. That is the worst
  failure available: indistinguishable from success. Every request must carry it
  as an explicit literal.
* **The train's date, not today's.** A historical run processes July dates in
  August. Settling "today" would regroup the wrong day, or nothing.
* **It must never fail a train.** A delivered train is delivered; settlement is
  re-runnable. Every exception is swallowed and logged.

No network: `requests` is a stub that records what it was asked to send.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import constants as C                                   # noqa: E402
from delivery import rake_settle as RS                            # noqa: E402


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _Requests:
    """Records calls; never touches the network."""

    def __init__(self, resp=None, raises=None):
        self.calls = []
        self._resp = resp or _Resp()
        self._raises = raises

    def post(self, url, params=None, timeout=None, **kw):
        self.calls.append({"url": url, "params": dict(params or {}),
                           "timeout": timeout})
        if self._raises:
            raise self._raises
        return self._resp


def _on(**kw):
    return RS.SettleConfig(enabled=True, url=C.SETTLE_API_URL, **kw)


# ---------------------------------------------------------------------------
# 1. The date comes from the batch key
# ---------------------------------------------------------------------------

class TestDateResolution(unittest.TestCase):

    def test_a_batch_key_yields_its_own_date(self):
        self.assertEqual(RS.batch_key_date("20260726_071641"), "2026-07-26")
        self.assertEqual(RS.batch_key_date("20260731_175009"), "2026-07-31")

    def test_a_non_batch_key_yields_none(self):
        for bad in ("", "RIGHT_UP", "2026-07-26", "20260726", "x_071641", None):
            self.assertIsNone(RS.batch_key_date(bad))

    def test_a_bad_key_refuses_rather_than_settling_some_other_day(self):
        req = _Requests()
        res = RS.settle_batch(batch_key="RIGHT_UP", cfg=_on(),
                              requests_mod=req, verbose=False)
        self.assertFalse(res.attempted)
        self.assertEqual(req.calls, [], "it sent a request with a guessed date")
        self.assertIn("refusing to guess", res.skipped_reason)

    def test_the_date_sent_is_the_trains_not_todays(self):
        req = _Requests()
        RS.settle_batch(batch_key="20260726_071641", cfg=_on(),
                        requests_mod=req, verbose=False)
        p = req.calls[0]["params"]
        self.assertEqual(p["start_date"], "2026-07-26")
        self.assertEqual(p["end_date"], "2026-07-26")


# ---------------------------------------------------------------------------
# 2. dry_run -- the trap
# ---------------------------------------------------------------------------

class TestDryRunIsAlwaysExplicit(unittest.TestCase):

    def test_a_real_settle_sends_dry_run_false(self):
        """Omitting it would preview, return 200, and write nothing."""
        req = _Requests()
        RS.settle_batch(batch_key="20260726_071641", cfg=_on(dry_run=False),
                        requests_mod=req, verbose=False)
        self.assertEqual(req.calls[0]["params"]["dry_run"], "false")

    def test_dry_run_is_never_merely_absent(self):
        for dry in (True, False):
            req = _Requests()
            RS.settle_batch(batch_key="20260726_071641",
                            cfg=_on(dry_run=dry), requests_mod=req,
                            verbose=False)
            self.assertIn("dry_run", req.calls[0]["params"],
                          "dry_run absent -> the API defaults it to TRUE")

    def test_dry_run_true_is_sent_and_reported_as_such(self):
        req = _Requests()
        res = RS.settle_batch(batch_key="20260726_071641",
                              cfg=_on(dry_run=True), requests_mod=req,
                              verbose=False)
        self.assertEqual(req.calls[0]["params"]["dry_run"], "true")
        self.assertIn("DRY RUN", res.render())
        self.assertIn("nothing written", res.render())

    def test_a_committed_settle_does_not_claim_to_be_a_dry_run(self):
        res = RS.settle_batch(batch_key="20260726_071641",
                              cfg=_on(dry_run=False),
                              requests_mod=_Requests(), verbose=False)
        self.assertIn("committed", res.render())
        self.assertNotIn("DRY RUN", res.render())


# ---------------------------------------------------------------------------
# 3. Off by default
# ---------------------------------------------------------------------------

class TestDisabledByDefault(unittest.TestCase):

    def test_the_default_config_is_off(self):
        self.assertFalse(RS.SettleConfig().enabled)

    def test_disabled_sends_nothing(self):
        req = _Requests()
        res = RS.settle_batch(batch_key="20260726_071641",
                              cfg=RS.SettleConfig(enabled=False),
                              requests_mod=req, verbose=False)
        self.assertEqual(req.calls, [])
        self.assertFalse(res.attempted)
        self.assertIn("WAGONEYE_AUTO_SETTLE", res.skipped_reason)

    def test_from_env_is_off_unless_asked(self):
        saved = os.environ.pop("WAGONEYE_AUTO_SETTLE", None)
        try:
            self.assertFalse(RS.SettleConfig.from_env().enabled)
            os.environ["WAGONEYE_AUTO_SETTLE"] = "1"
            self.assertTrue(RS.SettleConfig.from_env().enabled)
        finally:
            os.environ.pop("WAGONEYE_AUTO_SETTLE", None)
            if saved is not None:
                os.environ["WAGONEYE_AUTO_SETTLE"] = saved

    def test_from_env_reads_dry_run_and_defaults_it_to_a_real_settle(self):
        saved = os.environ.pop("WAGONEYE_SETTLE_DRY_RUN", None)
        try:
            self.assertFalse(RS.SettleConfig.from_env().dry_run)
            os.environ["WAGONEYE_SETTLE_DRY_RUN"] = "1"
            self.assertTrue(RS.SettleConfig.from_env().dry_run)
        finally:
            os.environ.pop("WAGONEYE_SETTLE_DRY_RUN", None)
            if saved is not None:
                os.environ["WAGONEYE_SETTLE_DRY_RUN"] = saved


# ---------------------------------------------------------------------------
# 4. It can never fail a train
# ---------------------------------------------------------------------------

class TestItNeverFailsATrain(unittest.TestCase):

    def test_a_connection_error_is_swallowed(self):
        res = RS.settle_batch(
            batch_key="20260726_071641", cfg=_on(),
            requests_mod=_Requests(raises=OSError("connection refused")),
            verbose=False)
        self.assertFalse(res.ok)
        self.assertIn("connection refused", res.error)

    def test_a_non_2xx_is_recorded_not_raised(self):
        res = RS.settle_batch(
            batch_key="20260726_071641", cfg=_on(),
            requests_mod=_Requests(_Resp(500, text="upstream boom")),
            verbose=False)
        self.assertFalse(res.ok)
        self.assertEqual(res.status_code, 500)
        self.assertIn("upstream boom", res.error)

    def test_a_failure_prints_the_retry_url_and_says_delivery_stands(self):
        res = RS.settle_batch(
            batch_key="20260726_071641", cfg=_on(),
            requests_mod=_Requests(_Resp(503)), verbose=False)
        line = res.render()
        self.assertIn("still delivered", line)
        self.assertIn("start_date=2026-07-26", line)
        self.assertIn("dry_run=false", line)

    def test_unparseable_json_is_still_a_success(self):
        """2xx means it ran. The body is for reporting, not for the verdict."""
        res = RS.settle_batch(
            batch_key="20260726_071641", cfg=_on(),
            requests_mod=_Requests(_Resp(200, payload=None)), verbose=False)
        self.assertTrue(res.ok)

    def test_no_input_makes_it_raise(self):
        for kw in ({"batch_key": ""}, {"batch_key": None}):
            RS.settle_batch(cfg=_on(), requests_mod=_Requests(),
                            verbose=False, **kw)   # must not raise


# ---------------------------------------------------------------------------
# 5. Endpoint + wiring
# ---------------------------------------------------------------------------

class TestEndpointAndWiring(unittest.TestCase):

    def test_the_configured_endpoint_is_the_settle_endpoint(self):
        self.assertTrue(C.SETTLE_API_URL.endswith("/cctv-watcher/settle"))

    def test_it_is_not_the_ingest_endpoint(self):
        """Settle cannot ingest; conflating them would take ingestion down."""
        self.assertNotEqual(C.SETTLE_API_URL, C.INGEST_API_URL_PROD)
        self.assertNotIn("inspections/ingest", C.SETTLE_API_URL)

    def test_the_url_is_overridable(self):
        req = _Requests()
        RS.settle_batch(batch_key="20260726_071641",
                        cfg=RS.SettleConfig(enabled=True,
                                            url="https://example/settle"),
                        requests_mod=req, verbose=False)
        self.assertEqual(req.calls[0]["url"], "https://example/settle")

    def test_a_timeout_is_always_sent(self):
        req = _Requests()
        RS.settle_batch(batch_key="20260726_071641", cfg=_on(),
                        requests_mod=req, verbose=False)
        self.assertIsNotNone(req.calls[0]["timeout"])

    def test_both_historical_branches_settle_only_when_delivering(self):
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        self.assertEqual(src.count("RS.settle_batch"), 2,
                         "sequential and batch must both settle")
        # Every call site sits under a `deliver` gate: a non-delivering run
        # reaches no external endpoint at all, and that must stay true.
        for chunk in src.split("RS.settle_batch")[:-1]:
            self.assertIn("if deliver:", chunk.split("if ok:")[-1])

    def test_historical_runner_still_reads_no_env_directly(self):
        """The invariant that stops it redefining an existing variable."""
        src = open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "orchestrator", "historical_runner.py"), encoding="utf-8").read()
        self.assertNotIn("os.getenv", src)
