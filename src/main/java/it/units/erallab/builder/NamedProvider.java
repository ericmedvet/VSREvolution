package it.units.erallab.builder;

import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author "Eric Medvet" on 2022/02/14 for VSREvolution
 */
public interface NamedProvider<T> {

  String TOKEN_SEPARATOR = ";";
  String PARAM_VALUE_SEPARATOR = "=";
  String NAME_KEY = "NAME";

  T build(Map<String, String> params);

  static <T> NamedProvider<T> empty() {
    return params -> {
      throw new NoSuchElementException();
    };
  }

  static <T> NamedProvider<T> of(Map<String, NamedProvider<? extends T>> providers) {
    return params -> providers.get(params.get(params.get(NAME_KEY))).build(params);
  }

  default NamedProvider<T> and(NamedProvider<? extends T> other) {
    NamedProvider<T> thisProvider = this;
    return params -> {
      try {
        return thisProvider.build(params);
      } catch (Throwable throwable) {
        return other.build(params);
      }
    };
  }

  default Optional<T> build() {
    return build("");
  }

  default Optional<T> build(String stringParams) {
    Map<String, String> params = Arrays.stream(stringParams.split(TOKEN_SEPARATOR))
        .map(s -> s.split(PARAM_VALUE_SEPARATOR))
        .collect(Collectors.toMap(
            ss -> ss.length == 2 ? ss[0] : NAME_KEY,
            ss -> ss.length == 2 ? ss[1] : ss[0]
        ));
    T t;
    try {
      t = build(params);
    } catch (Throwable throwable) {
      return Optional.empty();
    }
    return Optional.of(t);
  }

  default <R> NamedProvider<R> then(Function<? super T, ? extends R> function) {
    NamedProvider<T> thisProvider = this;
    return params -> function.apply(thisProvider.build(params));
  }

}
