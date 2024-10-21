from typing import List

from fastapi import APIRouter, HTTPException, WebSocket
from pydantic_models import VerificationRequest, VerificationResponse, Dataset, Example
from core import pipeline
from core import datasets
from starlette.websockets import WebSocketDisconnect
import json

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    is_connected = True

    try:
        while True:
            try:
                data = await websocket.receive_text()
                request = json.loads(data)

                async def progress_callback(message: str):
                    if is_connected:
                        try:
                            await websocket.send_text(json.dumps({"type": "progress", "message": message}))
                        except RuntimeError:
                            print("WebSocket unexpectedly closed. Stopping progress updates.")
                            return

                pipeline.set_progress_callback(progress_callback)

                result = await pipeline.verify(
                    request["word"],
                    request["claim"],
                    request.get("search_word"),
                    True
                )

                if is_connected:
                    try:
                        await websocket.send_text(json.dumps({"type": "result", "data": result}))
                    except RuntimeError:
                        print("WebSocket closed during result sending. Exiting.")
                        break

            except WebSocketDisconnect:
                print("Client disconnected")
                is_connected = False
                break
            except Exception as e:
                if is_connected:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                break

    finally:
        if is_connected:
            try:
                await websocket.close()
                is_connected = False
            except RuntimeError:
                print("Attempted to close an already closed WebSocket.")
            except Exception as e:
                print(f"Error while closing WebSocket: {str(e)}")


@router.post("/verify", response_model=VerificationResponse)
async def verify_claim(request: VerificationRequest):
    try:
        result = await pipeline.verify(request.word, request.claim, request.search_word)
        return VerificationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", response_model=List[Dataset])
async def get_datasets():
    try:
        response_data = [{"id": d["id"], "name": d["name"], "lang": d["lang"]} for d in datasets]
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/examples", response_model=List[Example])
async def get_dataset_examples(dataset_id: int):
    try:
        dataset = next((d for d in datasets if d["id"] == dataset_id), None)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if "examples" in dataset and dataset["examples"]:
            return dataset["examples"]
        else:
            raise HTTPException(status_code=404, detail="No examples available for this dataset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
