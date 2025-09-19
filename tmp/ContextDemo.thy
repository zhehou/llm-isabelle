theory ContextDemo imports Main begin

definition map_compose where "map (f ∘ g) xs = map f (map g xs)"

lemma append_assoc: "(xs @ ys) @ zs = xs @ (ys @ zs)"
lemma map_append: "map f (xs @ ys) = map f xs @ map f ys"
lemma rev_rev: "rev (rev xs) = xs"

end
