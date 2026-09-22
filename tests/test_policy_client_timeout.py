"""A missing policy server must not block evaluation indefinitely."""
import os
os.environ['RLDX_SKIP_HF_REGISTRATION'] = '1'
import socket
import time
import unittest
from rldx.policy.server_client import PolicyClient

class ClientTimeoutTest(unittest.TestCase):
    def test_missing_server_times_out_and_socket_can_be_recreated(self):
        with socket.socket() as port_reservation:
            port_reservation.bind(('127.0.0.1', 0))
            client = PolicyClient(port=port_reservation.getsockname()[1], timeout_ms=100)
            started = time.monotonic()
            self.assertFalse(client.ping())
            self.assertFalse(client.ping())
            self.assertLess(time.monotonic() - started, 3)
            client.socket.close(linger=0)
            client.context.term()

if __name__ == '__main__':
    unittest.main()
