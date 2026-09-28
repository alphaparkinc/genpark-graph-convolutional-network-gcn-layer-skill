import sys
import json
from client import GCNLayer

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-graph-convolutional-network-gcn-layer-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "gcn_forward_pass",
                        "description": "Execute spectral GCN layer forward pass: Z = ReLU(D_hat^-0.5 * A_hat * D_hat^-0.5 * X * W)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "adjacency": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "features": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                                "weights": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
                            },
                            "required": ["adjacency", "features", "weights"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "gcn_forward_pass":
            a = args.get("adjacency", [])
            x = args.get("features", [])
            w = args.get("weights", [])
            z = GCNLayer.forward(a, x, w)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"output_embeddings": z})}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
