import phoenix as px
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPTraceExporter
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

def setup_phoenix():
    """Start Phoenix and instrument LlamaIndex."""
    # Launch Phoenix locally
    session = px.launch_app()
    print(f"Phoenix Trace Viewer active at: {session.url}")

    # Set up the tracer and exporter
    endpoint = "http://localhost:6006/v1/traces"
    exporter = OTLPTraceExporter(endpoint=endpoint)
    tracer_provider = sdk_trace.TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(tracer_provider)

    # Instrument LlamaIndex
    LlamaIndexInstrumentor().instrument()
    print("LlamaIndex instrumentation active.")

if __name__ == "__main__":
    setup_phoenix()
    # Keep the script running to maintain the Phoenix session
    import time
    while True:
        time.sleep(10)
