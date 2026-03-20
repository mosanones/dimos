// Minimal LCM UDP multicast transport for the interop example.
// Implements only small-message encode/decode (no fragmentation).

use byteorder::{BigEndian, ByteOrder};
use socket2::{Domain, Protocol, Socket, Type};
use std::io;
use std::mem::MaybeUninit;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::sync::atomic::{AtomicU32, Ordering};

const MAGIC_SHORT: u32 = 0x4c433032; // "LC02"
const SHORT_HEADER_SIZE: usize = 8;
const LCM_MULTICAST_ADDR: Ipv4Addr = Ipv4Addr::new(239, 255, 76, 67);
const LCM_PORT: u16 = 7667;

static SEQ: AtomicU32 = AtomicU32::new(0);

pub struct LcmUdp {
    socket: Socket,
    multicast_addr: SocketAddrV4,
}

pub struct ReceivedMessage {
    pub channel: String,
    pub data: Vec<u8>,
}

impl LcmUdp {
    pub fn new() -> io::Result<Self> {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        #[cfg(not(target_os = "windows"))]
        socket.set_reuse_port(true)?;
        socket.set_nonblocking(true)?;

        let bind_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, LCM_PORT);
        socket.bind(&bind_addr.into())?;
        socket.join_multicast_v4(&LCM_MULTICAST_ADDR, &Ipv4Addr::UNSPECIFIED)?;
        socket.set_multicast_ttl_v4(1)?;

        Ok(Self {
            socket,
            multicast_addr: SocketAddrV4::new(LCM_MULTICAST_ADDR, LCM_PORT),
        })
    }

    /// Publish encoded LCM message data on the given channel.
    pub fn publish(&self, channel: &str, data: &[u8]) -> io::Result<()> {
        let channel_bytes = channel.as_bytes();
        let total = SHORT_HEADER_SIZE + channel_bytes.len() + 1 + data.len();
        let mut buf = vec![0u8; total];

        BigEndian::write_u32(&mut buf[0..4], MAGIC_SHORT);
        BigEndian::write_u32(&mut buf[4..8], SEQ.fetch_add(1, Ordering::Relaxed));

        buf[SHORT_HEADER_SIZE..SHORT_HEADER_SIZE + channel_bytes.len()]
            .copy_from_slice(channel_bytes);
        // null terminator already 0 from vec![0u8; ..]
        let payload_start = SHORT_HEADER_SIZE + channel_bytes.len() + 1;
        buf[payload_start..].copy_from_slice(data);

        self.socket.send_to(&buf, &self.multicast_addr.into())?;
        Ok(())
    }

    /// Try to receive one LCM message (non-blocking). Returns None if no data available.
    pub fn try_recv(&self) -> io::Result<Option<ReceivedMessage>> {
        let mut buf = [MaybeUninit::<u8>::uninit(); 65536];
        match self.socket.recv(&mut buf) {
            Ok(n) => {
                // SAFETY: socket2::recv guarantees the first `n` bytes are initialized.
                let buf =
                    unsafe { &*(&buf[..n] as *const [MaybeUninit<u8>] as *const [u8]) };
                if n < SHORT_HEADER_SIZE {
                    return Ok(None);
                }
                let magic = BigEndian::read_u32(&buf[0..4]);
                if magic != MAGIC_SHORT {
                    return Ok(None); // skip fragmented messages
                }
                // Find null terminator for channel name
                let channel_start = SHORT_HEADER_SIZE;
                let channel_end = match buf[channel_start..].iter().position(|&b| b == 0) {
                    Some(pos) => channel_start + pos,
                    None => return Ok(None),
                };
                let channel =
                    String::from_utf8_lossy(&buf[channel_start..channel_end]).into_owned();
                let data_start = channel_end + 1;
                let data = buf[data_start..].to_vec();
                Ok(Some(ReceivedMessage { channel, data }))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}
