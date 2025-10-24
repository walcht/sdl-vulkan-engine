/*
 * This code is public domain. Feel free to use it for any purpose!
 */

/* for details on the SDL_* function callbacks, read this guide:
 * https://wiki.libsdl.org/SDL3/README-main-functions */
#define SDL_MAIN_USE_CALLBACKS 1

#include "VkEngine.hpp"

/* Holds application state - this is mainly used to avoid global pointers
 * scattered all over the place */
struct AppState {
  std::string title;
  std::unique_ptr<vkengine::VkEngine> vkEnginePtr = nullptr;
};

/* This will be called once before anything else. If you want to, you can assign
 * a pointer to *appstate, and this pointer will be made available to you in
 * later functions calls in their appstate parameter. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {

  std::string title = "SDLVulkanEngine";
  std::string app_identifier = "com.walcht.sdlvkengine";

  //////////////////////////////////////////////////////////////////////////////
  // SDL3 Window Setup
  //////////////////////////////////////////////////////////////////////////////

  /* we allocate the global app state struct. This will be passed to other SDL
   * API calls so that we avoid global variables. We could have also used
   * SDL_malloc, but since this is a C++ project, new will be used instead. */
  auto _appstate = new AppState{.title{title.c_str()}};

  //////////////////////////////////////////////////////////////////////////////
  // Engine Setup
  //////////////////////////////////////////////////////////////////////////////

  try {
    _appstate->vkEnginePtr =
        std::make_unique<vkengine::VkEngine>(title, app_identifier);
  } catch (const std::exception &e) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[error]: %s", e.what());
    return SDL_APP_FAILURE;
  } catch (...) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "[error] unknown error occured");
    return SDL_APP_FAILURE;
  }

  /* don't forget to overwrite the output parameter for the app state */
  *appstate = _appstate;

  return SDL_APP_CONTINUE;
}

/* This is called over and over, possibly at the refresh rate of the display or
 * some other metric that the platform dictates. This is where the heart of your
 * app runs. */
SDL_AppResult SDL_AppIterate(void *appstate) {
  /* we rely on C++ exceptions to propagate and cleanly exit the program */
  try {
  } catch (const std::exception &e) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "%s", e.what());
    return SDL_APP_FAILURE;
  }
  return SDL_APP_CONTINUE;
}

/* This will be called whenever an SDL event arrives. Your app should not call
 * SDL_PollEvent, SDL_PumpEvent, etc, as SDL will manage all this for you.
 * Return values are the same as from SDL_AppIterate(), so you can terminate in
 * response to SDL_EVENT_QUIT, etc. */
SDL_AppResult SDL_AppEvent(void *appstate, SDL_Event *event) {
  if (event->type == SDL_EVENT_QUIT) {
    return SDL_APP_SUCCESS; /* successfully exit the program */
  }
  return SDL_APP_CONTINUE; /* continue the loop */
}

/* This is called once before terminating the app--assuming the app isn't being
 * forcibly killed or crashed--as a last chance to clean up. After this returns,
 * SDL will call SDL_Quit so the app doesn't have to (but it's safe for the app
 * to call it, too). */
void SDL_AppQuit(void *appstate, SDL_AppResult result) {
  free((AppState *)appstate);
}
