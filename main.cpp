/*
 * This code is public domain. Feel free to use it for any purpose!
 */

/* for details on the SDL_* function callbacks, read this guide:
 * https://wiki.libsdl.org/SDL3/README-main-functions */
#define SDL_MAIN_USE_CALLBACKS 1

/* make VulkanHpp function calls accept structure parameters directly (easier
 * to map to the original Vulkan C API) */
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1

#include "VkEngine.hpp"
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_vulkan.h>
#include <cstdlib>
#include <memory>
#include <string>

/* Holds application state - this is mainly used to avoid global pointers
 * scattered all over the place */
struct AppState {
  std::string title;
  SDL_Window *window;
  SDL_Renderer *renderer;
  std::unique_ptr<VkEngine> vkEnginePtr;
};

/* This will be called once before anything else. If you want to, you can assign
 * a pointer to *appstate, and this pointer will be made available to you in
 * later functions calls in their appstate parameter. */
SDL_AppResult SDL_AppInit(void **appstate, int argc, char **argv) {
  /* we allocate the global app state struct. This will be passed to other SDL
   * API calls so that we avoid global variables. We could have also used
   * SDL_malloc, but since this is a C++ project, new will be used instead. */
  auto _appstate = new AppState{.title{"SDLVulkanEngine"}};

  SDL_SetAppMetadata(_appstate->title.c_str(), "1.0",
                     "com.walcht.sdlvulkanengine");

  /* initialize the video subsystem */
  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "SDL_Init failed. Reason: %s",
                 SDL_GetError());
    return SDL_APP_FAILURE;
  }

  if ((_appstate->window = SDL_CreateWindow(_appstate->title.c_str(), 640, 480,
                                            SDL_WINDOW_VULKAN)) == nullptr) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR, "SDL_CreateWindow failed. Reason: %s",
                 SDL_GetError());
    return SDL_APP_FAILURE;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Vulkan Instance Creation & Validation Layers Setup
  //////////////////////////////////////////////////////////////////////////////
  constexpr vk::ApplicationInfo app_info{
      .pApplicationName = "SDLVulkanEngine",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
  };
  /* get required Vulkan extensions by SDL3 - these are REQUIRED extensions, so
   * we fail if any of them is not available */
  uint32_t sdl_extensions_count = 0;
  const char *const *sdl_extensions =
      SDL_Vulkan_GetInstanceExtensions(&sdl_extensions_count);

  std::vector<char const *> val_layers{"VK_LAYER_KHRONOS_validation"};

  //////////////////////////////////////////////////////////////////////////////
  // Physical Device & Queue Families
  //////////////////////////////////////////////////////////////////////////////

  /* get available physical devices with Vulkan support */
  auto devices = m_Instance.enumeratePhysicalDevices();
  if (devices.empty()) {
    SDL_LogError(SDL_LOG_CATEGORY_ERROR,
                 "no devices available with Vulkan support");
    return SDL_APP_FAILURE;
  }

  /* not all devices are equal - we need to find the device that satisfies our
   * particular requirements/needs */

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
